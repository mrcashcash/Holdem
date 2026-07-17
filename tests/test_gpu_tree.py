"""Parity of the flattened GPU betting tree with AbstractHoldem mechanics."""

import random
import unittest

from backend.abstraction.actions import ActionAbstraction
from backend.abstraction.buckets import AbstractionConfig, CardAbstraction
from backend.solver.gpu.tree import DECISION, FOLD_NODE, SHOWDOWN, STREET_END, BettingTree, GpuActionConfig
from backend.solver.holdem import AbstractHoldem


def make_pair() -> tuple[BettingTree, AbstractHoldem]:
    """A GPU tree and a reference game sharing the exact same betting menu."""
    config = GpuActionConfig(
        preflop_fractions=(1.0,),
        postflop_fractions=(0.75,),
        max_raises_per_street=3,
        stack_bb=50.0,
    )
    tree = BettingTree(config)
    abstraction_config = AbstractionConfig(
        flop_buckets=4,
        turn_buckets=4,
        river_buckets=3,
        fit_samples_per_street=60,
        flop_scenarios=8,
        opponents_per_scenario=6,
        seed=12,
    )
    abstraction = CardAbstraction(config=abstraction_config).fit()
    actions = ActionAbstraction(
        preflop_fractions=config.preflop_fractions,
        flop_fractions=config.postflop_fractions,
        turn_fractions=config.postflop_fractions,
        river_fractions=config.postflop_fractions,
        max_raises_per_street=config.max_raises_per_street,
    )
    return tree, AbstractHoldem(abstraction, actions, stack_bb=config.stack_bb)


class GpuTreeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree, cls.game = make_pair()

    def test_random_lines_match_reference_mechanics(self) -> None:
        rng = random.Random(0)
        for _ in range(400):
            state = self.game.initial_state().sample_chance(rng)
            node = self.tree.root
            while True:
                # Fast-forward the flattened tree through street boundaries.
                while self.tree.kind[node] == STREET_END:
                    node = self.tree.children[node][0]
                if state.is_chance():
                    state = state.sample_chance(rng)
                    continue
                if state.is_terminal():
                    kind = self.tree.kind[node]
                    self.assertIn(kind, (FOLD_NODE, SHOWDOWN))
                    if kind == FOLD_NODE:
                        self.assertEqual(int(self.tree.fold_loser[node]), state.folded)
                        self.assertAlmostEqual(
                            float(self.tree.fold_loser_committed[node]),
                            state.committed[state.folded],
                            places=4,
                        )
                    else:
                        self.assertIsNone(state.folded)
                        self.assertAlmostEqual(
                            float(self.tree.matched_pot[node]), min(state.committed), places=4
                        )
                    break
                self.assertEqual(self.tree.kind[node], DECISION)
                self.assertEqual(int(self.tree.actor[node]), state.current_player())
                self.assertEqual(int(self.tree.street[node]), state.street)
                reference_menu = sorted(state.legal_actions())
                tree_menu = sorted(
                    action for action in range(self.tree.config.num_actions) if self.tree.legal[node][action]
                )
                self.assertEqual(tree_menu, reference_menu)
                action = rng.choice(reference_menu)
                state = state.child(action)
                node = int(self.tree.children[node][action])

    def test_every_decision_has_a_child_per_legal_action(self) -> None:
        for node in self.tree.decision_nodes():
            for action in range(self.tree.config.num_actions):
                if self.tree.legal[node][action]:
                    self.assertGreaterEqual(self.tree.children[node][action], 0)
                else:
                    self.assertEqual(self.tree.children[node][action], -1)

    def test_terminal_pots_are_bounded_by_stacks(self) -> None:
        showdowns = self.tree.kind == SHOWDOWN
        self.assertTrue((self.tree.matched_pot[showdowns] <= self.tree.config.stack_bb).all())
        self.assertTrue((self.tree.matched_pot[showdowns] >= 1.0).all())


if __name__ == "__main__":
    unittest.main()
