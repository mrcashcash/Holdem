"""Never-limp ruleset: the tree structurally cannot limp, calls-vs-raises
survive, and serving translation maps an opponent's limp into the tree."""

import unittest

import numpy as np

from backend.solver.gpu.tree import CHECK_CALL, DECISION, FOLD, BettingTree, GpuActionConfig


def config(**overrides) -> GpuActionConfig:
    base = dict(
        preflop_fractions=(0.75, 1.0, 1.5),
        postflop_fractions=(0.33, 0.66, 1.0, 1.5),
        max_raises_per_street=2,
        stack_bb=50.0,
        no_limp=True,
    )
    base.update(overrides)
    return GpuActionConfig(**base)


class NoLimpTreeTests(unittest.TestCase):
    def test_root_offers_raise_or_fold_only(self) -> None:
        tree = BettingTree(config())
        legal = tree.legal[tree.root]
        self.assertTrue(legal[FOLD])
        self.assertFalse(legal[CHECK_CALL], "limp branch must not exist")
        self.assertTrue(any(legal[3:]), "raises must exist")

    def test_calls_facing_raises_survive(self) -> None:
        # Every preflop decision node EXCEPT the root open must keep
        # check/call (BB defending vs a raise, calling 3-bets, ...).
        tree = BettingTree(config())
        preflop_decisions = np.flatnonzero((tree.kind == DECISION) & (tree.street == 0))
        non_root = [n for n in preflop_decisions if n != tree.root]
        with_call = sum(bool(tree.legal[n][CHECK_CALL]) for n in non_root)
        self.assertGreater(with_call, 0, "facing-a-raise call branches disappeared")

    def test_default_config_still_limps(self) -> None:
        tree = BettingTree(config(no_limp=False))
        self.assertTrue(tree.legal[tree.root][CHECK_CALL])

    def test_translation_maps_opponent_limp_to_smallest_raise(self) -> None:
        from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

        tree = BettingTree(config())
        agent = GpuBlueprintAgent.__new__(GpuBlueprintAgent)  # translation needs no weights
        agent.tree = tree
        import random

        action = agent._translate_event(tree.root, None, {"action": "call"}, random.Random(0), tree=tree)
        self.assertGreaterEqual(action, 3, "limp must map to a raise action")
        fractions = tree.config.fractions(0)
        legal_raises = [3 + i for i in range(len(fractions)) if tree.legal[tree.root][3 + i]]
        smallest = min(legal_raises, key=lambda a: fractions[a - 3])
        self.assertEqual(action, smallest)


if __name__ == "__main__":
    unittest.main()
