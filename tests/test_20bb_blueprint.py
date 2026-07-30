"""Native 20bb blueprint configuration and artifact-retention guards."""

import unittest
from types import SimpleNamespace

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.solver.gpu import train
from backend.solver.gpu.tree import CHECK_CALL, BettingTree


class Blueprint20bbTests(unittest.TestCase):
    def test_native_config_is_rich_and_shallow(self) -> None:
        config = train.BLUEPRINT_CONFIG_20
        self.assertEqual(config.stack_bb, 20.0)
        self.assertEqual(config.preflop_fractions, (0.5, 0.75))
        self.assertEqual(config.postflop_fractions, (0.33, 0.66, 1.0, 1.5))
        self.assertEqual(config.max_raises_per_street, 2)
        self.assertTrue(config.no_donk_srp)

    def test_native_tree_stays_inside_sizing_ceiling(self) -> None:
        tree = BettingTree(train.BLUEPRINT_CONFIG_20)
        self.assertLessEqual(len(tree), 200_000)
        self.assertEqual(len(tree), 36_906)

    def test_out_of_position_caller_cannot_donk_in_single_raised_pot(self) -> None:
        tree = BettingTree(train.BLUEPRINT_CONFIG_20)
        # Button/SB raises, BB calls, then BB is first to act on the flop.
        facing_open = tree.children[tree.root, 3]
        street_end = tree.children[facing_open, CHECK_CALL]
        flop = tree.children[street_end, 0]
        self.assertEqual(tree.actor[flop], 1)
        self.assertEqual(tree.legal[flop].nonzero()[0].tolist(), [CHECK_CALL])

    def test_out_of_position_aggressor_may_lead_after_limp_raise(self) -> None:
        tree = BettingTree(train.BLUEPRINT_CONFIG_20)
        # Button/SB limps, BB raises, SB calls; BB is both OOP and aggressor.
        bb_after_limp = tree.children[tree.root, CHECK_CALL]
        facing_raise = tree.children[bb_after_limp, 3]
        street_end = tree.children[facing_raise, CHECK_CALL]
        flop = tree.children[street_end, 0]
        self.assertEqual(tree.actor[flop], 1)
        self.assertGreater(int(tree.legal[flop].sum()), 1)

    def test_check_raiser_may_lead_the_next_street(self) -> None:
        tree = BettingTree(train.BLUEPRINT_CONFIG_20)
        facing_open = tree.children[tree.root, 3]
        street_end = tree.children[facing_open, CHECK_CALL]
        flop_oop = tree.children[street_end, 0]
        flop_ip = tree.children[flop_oop, CHECK_CALL]
        facing_bet = tree.children[flop_ip, 3]
        facing_check_raise = tree.children[facing_bet, 3]
        turn_end = tree.children[facing_check_raise, CHECK_CALL]
        turn_oop = tree.children[turn_end, 0]
        self.assertEqual(tree.actor[turn_oop], 1)
        self.assertGreater(int(tree.legal[turn_oop].sum()), 1)

    def test_early_gate_milestones_are_retained(self) -> None:
        self.assertTrue({5_000, 10_000, 20_000}.issubset(train.MILESTONE_ITERATIONS))

    def test_serving_guard_blocks_resolver_donk(self) -> None:
        game = SimpleNamespace(
            street=1,
            public_actions=[
                {"street": 0, "player": 0, "action": "raise"},
                {"street": 0, "player": 1, "action": "call"},
            ],
            legal_actions=lambda player: {"check": True, "raise": True},
        )
        self.assertTrue(GpuBlueprintAgent._must_check_no_donk(game, 1))
        self.assertFalse(GpuBlueprintAgent._must_check_no_donk(game, 0))

    def test_serving_guard_allows_last_aggressor_to_lead(self) -> None:
        game = SimpleNamespace(
            street=2,
            public_actions=[
                {"street": 0, "player": 0, "action": "raise"},
                {"street": 0, "player": 1, "action": "call"},
                {"street": 1, "player": 1, "action": "check"},
                {"street": 1, "player": 0, "action": "raise"},
                {"street": 1, "player": 1, "action": "raise"},
                {"street": 1, "player": 0, "action": "call"},
            ],
            legal_actions=lambda player: {"check": True, "raise": True},
        )
        self.assertFalse(GpuBlueprintAgent._must_check_no_donk(game, 1))


if __name__ == "__main__":
    unittest.main()
