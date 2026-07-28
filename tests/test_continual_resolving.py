"""Continual turn+river resolving with exact cards (P1.3).

Two things are pinned here.

**Chance resampling.** `_GadgetGraphRunner` captures a CUDA graph that reads its
deal from fixed buffers. For the river that is exact -- `ExactRiverSampler` has
exactly one possible deal. A TURN sampler has 48 river runouts, so replaying a
captured graph without refilling those buffers would solve the turn as if one
specific river card had already arrived. That is a silent, strategy-destroying
bug (the agent would "know" the river), and nothing about the output looks wrong,
so it gets a direct test.

**Range continuity.** The session must consult the blueprint exactly once, to
seed itself, and thereafter advance ranges from the policies actually played.
Re-deriving from the blueprint mid-hand is the self-range inconsistency that made
v1 search regress by 86 bb/100.
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path

import numpy as np

from backend.poker import HeadsUpHoldem
from backend.search.exact_turn import ExactTurnSampler
from backend.solver.gpu.deals import NUM_COMBOS

CHECKPOINTS = (
    Path("backend/data/gpu_blueprint_200bb/champion.npz"),
    Path("backend/data/gpu_blueprint/champion.npz"),
)
BOARD = (0, 17, 30, 43)


def _load_agent():
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    for path in CHECKPOINTS:
        if path.exists():
            agent = GpuBlueprintAgent.try_load(path)
            if agent is not None:
                agent.subgame_search = False
                agent.flop_search = False
                agent.exact_river_search = False
                return agent
    return None


class CountingSampler:
    """Wraps ExactTurnSampler and records which rivers were actually drawn."""

    def __init__(self, board: tuple[int, ...]) -> None:
        self.inner = ExactTurnSampler(board)
        self.drawn: list[int] = []

    def bucket_counts(self):
        return self.inner.bucket_counts()

    def sample(self, rng):
        deal = self.inner.sample(rng)
        self.drawn.append(deal.board[4])
        return deal


class ChanceResamplingTests(unittest.TestCase):
    """The captured graph must see fresh river cards, not one frozen runout."""

    def _gadget(self, sampler):
        import torch

        from backend.search.safe_subgame import GadgetCFR
        from backend.solver.gpu.cfr import VectorCFR
        from backend.solver.gpu.tree import (
            BettingRootState,
            BettingTree,
            GpuActionConfig,
        )

        if not torch.cuda.is_available():
            self.skipTest("graph capture requires CUDA")
        config = GpuActionConfig(
            preflop_fractions=(1.0,), postflop_fractions=(0.5, 1.0),
            max_raises_per_street=2, stack_bb=100.0,
        )
        root = BettingRootState(
            street=2, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
            stacks=(90.0, 90.0), acted=(False, False), raises=0, last_increment=1.0,
        )
        tree = BettingTree(config, root_state=root)
        solver = VectorCFR(tree, sampler, device="cuda", seed=5, averaging_delay=2)
        live = np.zeros((2, NUM_COMBOS), dtype=np.float32)
        mask = sampler.inner.deal_for_river(sampler.inner.rivers[0]).valid
        live[0] = mask / mask.sum()
        live[1] = mask / mask.sum()
        solver.root_reach = torch.as_tensor(live, device=solver.device)
        gadget = GadgetCFR(
            solver, constrained=0, base_ranges=live,
            alt=torch.zeros(NUM_COMBOS, device=solver.device),
        )
        return gadget

    def test_resampling_draws_many_distinct_rivers(self) -> None:
        import time

        from backend.search.exact_river import _run_gadget

        sampler = CountingSampler(BOARD)
        gadget = self._gadget(sampler)
        sampler.drawn.clear()
        _run_gadget(gadget, iterations=60, deadline=time.monotonic() + 120.0, resample=True)
        distinct = set(sampler.drawn)
        self.assertGreater(
            len(distinct), 10,
            f"resample=True drew only {len(distinct)} distinct rivers: the turn "
            "solve would be conditioned on a frozen runout",
        )

    def test_without_resampling_the_graph_freezes_one_river(self) -> None:
        """Documents WHY the flag exists (and that river reuse is deliberate)."""
        import time

        from backend.search.exact_river import _run_gadget

        sampler = CountingSampler(BOARD)
        gadget = self._gadget(sampler)
        sampler.drawn.clear()
        _run_gadget(gadget, iterations=60, deadline=time.monotonic() + 120.0, resample=False)
        # Capture itself draws a deal; the replay loop draws none.
        self.assertLessEqual(len(set(sampler.drawn)), 1)


class ContinualSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        agent = _load_agent()
        if agent is None:
            raise unittest.SkipTest("no blueprint checkpoint available")
        cls.agent = agent

    def _turn_engine(self, seed: int = 4) -> HeadsUpHoldem:
        """Play a passive line to the turn so a real history exists."""
        engine = HeadsUpHoldem(
            initial_stack=4000, small_blind=10, big_blind=20, rng=random.Random(seed)
        )
        guard = 0
        while engine.street < 2 and not engine.hand_complete and guard < 40:
            guard += 1
            player = engine.current_player
            legal = engine.legal_actions(player)
            engine.act(player, "check" if legal.get("check") else "call")
        return engine

    def test_session_seeds_once_and_carries_ranges_forward(self) -> None:
        from backend.search.continual import (
            ContinualSession,
            open_session,
            resolve_decision,
        )

        engine = self._turn_engine()
        if engine.street != 2 or engine.hand_complete:
            self.skipTest("scripted line did not reach the turn")

        sessions: dict[tuple[int, int], ContinualSession] = {}
        key = (1, engine.hand_number)
        session = open_session(self.agent, engine, engine.current_player, key, sessions)
        self.assertEqual(session.entry_street, 2)
        self.assertEqual(len(session.board), 4)
        # Seeded ranges are proper distributions over live combos.
        for seat in (0, 1):
            self.assertAlmostEqual(float(session.ranges[seat].sum()), 1.0, places=5)
            self.assertTrue(np.all(session.ranges[seat] >= 0.0))

        # Reopening returns the SAME object: the blueprint is consulted once.
        again = open_session(self.agent, engine, engine.current_player, key, sessions)
        self.assertIs(again, session)

        solution = resolve_decision(
            self.agent, engine, engine.current_player,
            key=key, sessions=sessions, iterations=24, budget_ms=60_000,
        )
        diagnostics = solution.diagnostics
        self.assertEqual(diagnostics["street"], 2)
        self.assertEqual(diagnostics["entry_street"], 2)
        self.assertEqual(diagnostics["mode"], "continual-exact-v1-street2")
        self.assertEqual(diagnostics["exact_private_combos"], NUM_COMBOS)
        self.assertGreaterEqual(diagnostics["iterations"], 12)
        self.assertEqual(diagnostics["river_runouts"], 48)

        # The root policy must be a valid distribution for live combos.
        tree = solution.tree
        row = solution.strategy[tree.root]
        legal = np.asarray(tree.legal[tree.root], dtype=bool)
        self.assertTrue(np.all(row[:, ~legal] == 0.0))
        live = session.ranges[0] + session.ranges[1] > 0
        np.testing.assert_allclose(row[live].sum(axis=1), 1.0, atol=1e-5)

    def test_survives_the_turn_to_river_transition_without_reseeding(self) -> None:
        """The river card must not reset the session to blueprint ranges."""
        from backend.search.continual import ContinualSession, open_session

        engine = self._turn_engine()
        if engine.street != 2:
            self.skipTest("scripted line did not reach the turn")
        sessions: dict[tuple[int, int], ContinualSession] = {}
        key = (2, engine.hand_number)
        session = open_session(self.agent, engine, engine.current_player, key, sessions)
        seeded = session.ranges.copy()

        # Advance to the river by checking it down.
        guard = 0
        while engine.street == 2 and not engine.hand_complete and guard < 10:
            guard += 1
            player = engine.current_player
            legal = engine.legal_actions(player)
            engine.act(player, "check" if legal.get("check") else "call")
        if engine.street != 3:
            self.skipTest("could not reach the river")

        same = open_session(self.agent, engine, engine.current_player, key, sessions)
        self.assertIs(same, session, "river card reseeded the session from the blueprint")
        self.assertEqual(len(same.board), 4, "session keeps its turn entry board")
        np.testing.assert_array_equal(same.ranges, seeded)

    def test_rejects_streets_it_does_not_cover(self) -> None:
        from backend.search.continual import ContinualResolveError, resolve_decision

        engine = HeadsUpHoldem(initial_stack=4000, small_blind=10, big_blind=20, rng=random.Random(1))
        with self.assertRaises(ContinualResolveError):
            resolve_decision(
                self.agent, engine, 0, key=(9, 1), sessions={},
                iterations=12, budget_ms=1000,
            )


if __name__ == "__main__":
    unittest.main()
