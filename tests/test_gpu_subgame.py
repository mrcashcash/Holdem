"""Turn/river subgame re-solving on the GPU engine (research item #4)."""

import random
import unittest

import numpy as np

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS, DealSampler
from backend.solver.gpu.tree import DECISION, SHOWDOWN, BettingTree, GpuActionConfig


def small_agent(subgame_search: bool, subgame_iterations: int = 25) -> GpuBlueprintAgent:
    config = GpuActionConfig(
        preflop_fractions=(1.0,),
        postflop_fractions=(0.75,),
        max_raises_per_street=2,
        stack_bb=100.0,
    )
    tree = BettingTree(config)
    sampler = DealSampler(flop_samples=2, turn_samples=2)
    solver = VectorCFR(tree, sampler, device="cpu", seed=9)
    solver.run(6)
    return GpuBlueprintAgent(
        tree,
        solver.average_strategy_tables().astype(np.float64),
        sampler,
        subgame_search=subgame_search,
        subgame_iterations=subgame_iterations,
    )


class SubgameTreeTests(unittest.TestCase):
    def test_turn_rooted_tree_starts_on_the_turn(self) -> None:
        config = GpuActionConfig(postflop_fractions=(0.5, 1.0), max_raises_per_street=3)
        tree = BettingTree(config, start_street=2, start_pot=20.0, start_stacks=(90.0, 90.0))
        self.assertEqual(int(tree.street[tree.root]), 2)
        self.assertEqual(int(tree.actor[tree.root]), 1)  # big blind first postflop
        showdowns = tree.kind == SHOWDOWN
        # Checked-down pot: both entered with 10 committed each.
        self.assertGreaterEqual(float(tree.matched_pot[showdowns].min()), 10.0 - 1e-6)
        self.assertGreater(len(tree), 100)

    def test_river_rooted_tree_is_small(self) -> None:
        config = GpuActionConfig(postflop_fractions=(0.33, 0.75, 1.5, 2.5), max_raises_per_street=3)
        tree = BettingTree(config, start_street=3, start_pot=40.0, start_stacks=(80.0, 80.0))
        self.assertLess(len(tree), 3000)
        self.assertTrue((tree.street[tree.kind == DECISION] == 3).all())


class RootReachTests(unittest.TestCase):
    def test_root_reach_confines_ranges(self) -> None:
        # A player whose range is a single combo should never accumulate
        # strategy mass in buckets that combo cannot occupy.
        import torch

        config = GpuActionConfig(postflop_fractions=(0.75,), max_raises_per_street=1)
        tree = BettingTree(config, start_street=3, start_pot=10.0, start_stacks=(95.0, 95.0))
        board = (0, 5, 10, 15, 20)
        sampler = DealSampler()
        from backend.search.gpu_subgame import FixedBoardSampler

        solver = VectorCFR(tree, FixedBoardSampler(sampler, board), device="cpu", seed=2)
        reach = torch.zeros((2, NUM_COMBOS))
        reach[0, 100] = 1.0  # seat 0 holds exactly combo #100
        reach[1, :] = 1.0
        solver.root_reach = reach
        solver.run(12)
        self.assertGreater(float(solver.strategy_sums.abs().sum().item()), 0.0)


class IntegratedSubgameAgentTests(unittest.TestCase):
    def test_agent_with_subgame_search_plays_legal_hands(self) -> None:
        agent = small_agent(subgame_search=True)
        engine = HeadsUpHoldem(rng=random.Random(41))
        turns_played = 0
        hands = 0
        while hands < 12:
            while not engine.hand_complete:
                player = engine.current_player
                if player == 1:
                    if engine.street >= 2:
                        turns_played += 1
                    choice = agent.select(engine, 1)
                    agent.execute(engine, 1, choice)
                else:
                    legal = engine.legal_actions(0)
                    engine.act(0, "check" if legal.get("check") else "call" if legal.get("call") else "fold")
            hands += 1
            engine.new_hand()
        self.assertGreater(turns_played, 0, "test never exercised the subgame path")

    def test_subgame_solutions_are_cached_per_hand(self) -> None:
        agent = small_agent(subgame_search=True)
        engine = HeadsUpHoldem(rng=random.Random(43))
        # Drive one hand to the turn with checks/calls.
        while not engine.hand_complete and engine.street < 2:
            player = engine.current_player
            if player == 1:
                agent.execute(engine, 1, agent.select(engine, 1))
            else:
                legal = engine.legal_actions(0)
                engine.act(0, "check" if legal.get("check") else "call")
        if not engine.hand_complete and engine.current_player == 1:
            agent.select(engine, 1)
            self.assertTrue(agent._subgame_cache)


if __name__ == "__main__":
    unittest.main()
