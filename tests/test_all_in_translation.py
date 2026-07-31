"""The translated ALL-IN must not become a bet the blueprint never chose.

Regression cover for a live hand reported 2026-07-28: on T(c)J(s)K(h)A(c) the
agent moved in for 3,980 chips into a 166-chip pot -- 24x the pot -- from a
blueprint whose widest own-bet size is 1.0x pot.

Cause (measured, `tools/overbet_audit.py`): `_locate` matches the real hand onto
an abstract node by translated action sequence alone and never compares pot/stack
geometry. Every `raise` action re-derives its chip amount from the REAL pot, but
ALL-IN used to execute literally, so it was the one size-bearing action whose
meaning translation could silently destroy.

The arithmetic tests use stubs so each expected amount is exact rather than
approximately plausible; the end-to-end test then plays real hands through the
real serving agent, because a guard that is correct in isolation and unreachable
in the serving path fixes nothing.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from backend.agents.gpu_blueprint_agent import (
    ALL_IN_MAX_POT_MULTIPLE,
    GpuBlueprintAgent,
)

BIG_BLIND = 20


def make_blueprint(stack_bb: float, abstract_pot_bb: float, guard: bool = True):
    """Minimal stand-in exposing exactly what `_all_in_size` reads."""
    stub = SimpleNamespace(
        tree=SimpleNamespace(config=SimpleNamespace(stack_bb=stack_bb)),
        all_in_geometry_guard=guard,
    )
    stub._abstract_matched_pot = lambda node: abstract_pot_bb
    stub._all_in_size = lambda game, player, node: GpuBlueprintAgent._all_in_size(
        stub, game, player, node
    )
    return stub


def make_game(*, contributions: list[int], stacks: list[int], round_bets: list[int] | None = None):
    """A postflop state with both players level and nobody facing a bet."""
    bets = list(round_bets if round_bets is not None else [0, 0])
    return SimpleNamespace(
        contributions=list(contributions),
        stacks=list(stacks),
        round_bets=bets,
        legal_actions=lambda player: {
            "all_in": True,
            "raise": True,
            "raise_min": max(BIG_BLIND, max(bets) * 2),
            "raise_max": bets[player] + stacks[player],
            "to_call": max(bets) - bets[player],
        },
    )


class AllInGeometryTests(unittest.TestCase):
    def test_matching_geometry_leaves_the_jam_alone(self) -> None:
        """When the abstract and real pots agree, ALL-IN keeps its trained meaning."""
        # 40bb stack, matched pot 20bb abstractly and really: a 2x-pot jam.
        blueprint = make_blueprint(stack_bb=40.0, abstract_pot_bb=20.0)
        game = make_game(contributions=[200, 200], stacks=[600, 600])
        self.assertIsNone(blueprint._all_in_size(game, 0, 7))

    def test_short_stack_still_jams(self) -> None:
        """A genuine short-stack shove is not an overbet and must survive untouched."""
        blueprint = make_blueprint(stack_bb=20.0, abstract_pot_bb=8.0)
        # 20bb start, 4bb in each: 16bb behind into an 8bb pot = 2x pot.
        game = make_game(contributions=[80, 80], stacks=[320, 320])
        self.assertIsNone(blueprint._all_in_size(game, 0, 3))

    def test_reported_hand_is_resized_and_bounded(self) -> None:
        """The 24x-pot shove becomes a bet bounded by the absolute cap."""
        # The live hand: 199bb behind, matched pot 166 chips (8.3bb).
        blueprint = make_blueprint(stack_bb=200.0, abstract_pot_bb=60.0)
        game = make_game(contributions=[83, 83], stacks=[3_980, 3_980])
        sized = blueprint._all_in_size(game, 0, 104_350)
        self.assertIsNotNone(sized, "a 24x-pot jam must be resized")
        target, diagnostics = sized
        matched_real = 2 * min(game.contributions)
        self.assertLess(target, game.stacks[0], "resized bet must not still be the stack")
        self.assertLessEqual(target, ALL_IN_MAX_POT_MULTIPLE * matched_real + 1)
        self.assertGreater(
            diagnostics["real_shove_x_pot"], diagnostics["abstract_shove_x_pot"]
        )

    def test_resized_amount_is_exact(self) -> None:
        """The bet reproduces the abstract jam's pot multiple, to the chip."""
        # Abstract: 100bb stack over a 25bb matched pot = a 4x-pot jam.
        blueprint = make_blueprint(stack_bb=100.0, abstract_pot_bb=25.0)
        # Real: matched pot 200 chips, 3,000 behind = a 15x-pot jam. Distorted,
        # so the guard wants 4x the real matched pot = 800 chips COMMITTED in
        # total. 100 of that is already in from earlier streets, and the target
        # is a raise-TO for the current street, so it asks for 700.
        game = make_game(contributions=[100, 100], stacks=[3_000, 3_000])
        sized = blueprint._all_in_size(game, 0, 1)
        self.assertIsNotNone(sized)
        self.assertEqual(sized[0], 700)

    def test_matched_geometry_survives_even_when_the_jam_is_huge(self) -> None:
        """A big jam that translated PERFECTLY must not be touched.

        This case previously asserted the opposite -- that a 20x-pot jam is
        trimmed even when the real geometry matches it -- because the cap was
        folded into the trigger as min(ratio_abstract * tolerance, cap). That
        made the geometry test unreachable for any abstract jam above 6x pot.
        Measured over 150 LBR pairs at 200bb, 89.6% of all firings were this
        false positive (483 of them with real == abstract exactly), which is the
        real reason the guard measured -268.82 bb/100 rather than any
        GTO-versus-exploitation tension.

        Abstract: 200bb stack over a 10bb matched pot = a 20x-pot jam.
        Real: 3,900 chips over a 200-chip matched pot = 19.5x. That agrees, so
        there is nothing to correct.
        """
        blueprint = make_blueprint(stack_bb=200.0, abstract_pot_bb=10.0)
        game = make_game(contributions=[100, 100], stacks=[3_800, 3_800])
        self.assertIsNone(blueprint._all_in_size(game, 0, 2))

    def test_preflop_jam_is_never_trimmed(self) -> None:
        """A preflop all-in is inherently ~100-200x the matched pot. That is not a leak.

        The regression this locks down: at 200bb the matched preflop pot is one
        big blind, so every preflop shove looks like a 100x+ "overbet" to an
        absolute pot-multiple bound, and the old trigger trimmed it to 6x pot --
        converting an all-in into a small raise.
        """
        blueprint = make_blueprint(stack_bb=200.0, abstract_pot_bb=2.0)
        # Matched pot 40 chips (2bb); 4,000 committed if shoving = 100x pot,
        # exactly what the abstract jam also represents.
        game = make_game(contributions=[20, 20], stacks=[3_980, 3_980])
        self.assertIsNone(blueprint._all_in_size(game, 0, 11))

    def test_cap_bounds_the_correction_when_distortion_is_genuine(self) -> None:
        """Where geometry really is distorted, the cap still bounds the repair."""
        # Abstract: 200bb over a 10bb matched pot = 20x. Real: 7,000 over 200 =
        # 35x, which exceeds 20 * 1.5, so this is genuine distortion.
        blueprint = make_blueprint(stack_bb=200.0, abstract_pot_bb=10.0)
        game = make_game(contributions=[100, 100], stacks=[6_900, 6_900])
        sized = blueprint._all_in_size(game, 0, 2)
        self.assertIsNotNone(sized, "a 35x-pot jam against a 20x abstract jam is distorted")
        # The correction wants min(20, 6) = 6x the 200-chip matched pot = 1,200
        # committed, less the 100 already in from earlier streets.
        self.assertAlmostEqual(sized[0], ALL_IN_MAX_POT_MULTIPLE * 200 - 100, delta=1)

    def test_guard_can_be_disabled(self) -> None:
        """The A/B switch really disables the guard, or its measurement is a lie."""
        blueprint = make_blueprint(stack_bb=200.0, abstract_pot_bb=60.0, guard=False)
        game = make_game(contributions=[83, 83], stacks=[3_980, 3_980])
        self.assertIsNone(blueprint._all_in_size(game, 0, 104_350))


class ServingPathTests(unittest.TestCase):
    """End-to-end behaviour through the real serving agent.

    A min-raising opponent on purpose: that is the line which exhausts the tree's
    raise cap, and it was the only one of three opponents to provoke overbets at
    all -- a calling station and self-play produced zero.
    """

    def _worst_jam(self, guard: bool) -> tuple[float, list[tuple]]:
        from backend.agents.serving import load_serving_agent
        from backend.eval.null_agents import ScriptedAgent
        from backend.poker import HeadsUpHoldem

        agent = load_serving_agent()
        if not hasattr(agent, "all_in_geometry_guard"):
            raise unittest.SkipTest("no blueprint checkpoint available to serve")
        agent.continual_search = False  # isolate the guard from the resolver
        agent.all_in_geometry_guard = guard
        villain = ScriptedAgent("always-min-raise")

        stack = 200 * BIG_BLIND
        engine = HeadsUpHoldem(
            initial_stack=stack, small_blind=BIG_BLIND // 2, big_blind=BIG_BLIND,
            rng=random.Random(20260728),
        )
        worst = 0.0
        offenders: list[tuple] = []
        for _ in range(120):
            engine.stacks = [stack, stack]
            engine.new_hand()
            while not engine.hand_complete:
                seat = engine.current_player
                if seat != 0:
                    villain.execute(engine, seat, villain.select(engine, seat))
                    continue
                legal = engine.legal_actions(0)
                matched = 2 * min(engine.contributions)
                had_choice = (
                    bool(legal.get("raise")) and legal["raise_max"] > legal["raise_min"]
                )
                agent.execute(engine, 0, agent.select(engine, 0))
                event = engine.public_actions[-1]
                amount = int(event.get("amount", 0) or 0)
                if had_choice and amount and matched:
                    ratio = amount / matched
                    worst = max(worst, ratio)
                    if ratio > ALL_IN_MAX_POT_MULTIPLE + 1.0:
                        offenders.append(
                            (event.get("street"), matched, amount, round(ratio, 1))
                        )
        return worst, offenders

    def test_guard_bounds_postflop_jams_but_leaves_preflop_alone(self) -> None:
        """With the guard on, POSTFLOP jams are bounded; preflop shoves are not.

        This previously asserted that NO jam may exceed the cap on any street.
        That invariant was the absolute-pot-multiple bound, and it is wrong
        preflop: the matched preflop pot is a blind or two, so a 200bb shove is
        inherently 10-100x pot. Trimming it converts an all-in into a small
        raise. The distortion the guard exists to fix comes from raise-cap
        exhaustion, which is a POSTFLOP phenomenon -- so that is where the bound
        belongs.
        """
        worst, offenders = self._worst_jam(guard=True)
        postflop = [entry for entry in offenders if entry[0] != 0]
        self.assertFalse(
            postflop,
            f"postflop jams beyond {ALL_IN_MAX_POT_MULTIPLE}x the matched pot: "
            f"{postflop[:5]} (worst overall {worst:.1f}x)",
        )

    def test_default_still_makes_the_huge_jams(self) -> None:
        """The shipped default does NOT bound the jam, and that is deliberate.

        Serving default is guard-off because the guard measured -268.82 bb/100 at
        200bb and -124.00 at 100bb against exactly this opponent: a station calls
        any jam, so shoving into it is correct exploitation. This test pins the
        consequence so nobody reads the guard's existence as it being active.
        """
        worst, offenders = self._worst_jam(guard=False)
        self.assertTrue(
            offenders,
            "expected unbounded jams with the guard off; if this now passes, the "
            "default changed or the abstraction did",
        )
        self.assertGreater(worst, ALL_IN_MAX_POT_MULTIPLE)


if __name__ == "__main__":
    unittest.main()
