"""Duel-harness NULL test: an agent dueling itself MUST read ~0.

Guards against baseline/accounting bias (2026-07-23: blinds posted at engine
construction inflated every duplicate pair by +(SB+BB)/2 = +75 bb/100, which
masqueraded as a consistent challenger edge across ten milestone gates)."""

import unittest

from backend.eval.duel import head_to_head
from backend.styles import HeuristicAgent


class DuelNullTests(unittest.TestCase):
    def test_stochastic_self_duel_reads_exactly_zero_under_crn(self) -> None:
        """The gap this suite previously left open.

        `HeuristicAgent` is deterministic, so its pair cancels exactly and this
        suite passed while the harness was in fact unusable for the STOCHASTIC
        blueprint agents every real gate compares. On 2026-07-27 an off-vs-off
        null of two blueprint copies read +34.91 bb/100 [+12.27, +57.55] —
        significant, from identical policies. Across seeds it was +49.10 / −18.65
        / +8.06: unbiased but far too noisy for a single-seed verdict.

        Common random numbers restores exact cancellation for stochastic agents.
        """
        import random

        class Mixed:
            """Minimal stochastic agent: a coin flip between two legal actions."""

            def __init__(self) -> None:
                self._rng = random.Random(97)

            def select(self, game, player):
                legal = game.legal_actions(player)
                if self._rng.random() < 0.5 and legal.get("raise"):
                    return 2
                return 1

            def execute(self, game, player, choice):
                legal = game.legal_actions(player)
                if choice == 2 and legal.get("raise"):
                    game.act(player, "raise", int(legal["raise_min"]))
                elif legal.get("check"):
                    game.act(player, "check")
                else:
                    game.act(player, "call")

        result = head_to_head(
            Mixed(), Mixed(), stack_bb=100.0, pairs=150, seed=11,
            common_random_numbers=True,
        )
        self.assertAlmostEqual(result["mean_bb_per_100"], 0.0, places=9, msg=result)
        self.assertEqual(result["verdict"], "KEEP")

    def test_self_duel_reads_zero(self) -> None:
        # A deterministic policy against itself: duplicate seat-swap makes the
        # pair cancel EXACTLY, so any nonzero mean is harness bias by
        # construction (the old bug read +75.0 here).
        agent = HeuristicAgent()
        result = head_to_head(agent, agent, stack_bb=100.0, pairs=120, seed=11)
        self.assertLess(abs(result["mean_bb_per_100"]), 1.0, f"harness bias: {result}")
        self.assertEqual(result["verdict"], "KEEP")


if __name__ == "__main__":
    unittest.main()
