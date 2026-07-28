"""Depth routing: each hand is served by the nearest-stack blueprint."""

import random
import unittest

import numpy as np

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.agents.multistack_agent import MultiStackBlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig


def tiny_agent(stack_bb: float) -> GpuBlueprintAgent:
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.75,), max_raises_per_street=2, stack_bb=stack_bb
    )
    tree = BettingTree(config)
    sampler = DealSampler(flop_samples=2, turn_samples=2)
    solver = VectorCFR(tree, sampler, device="cpu", seed=1)
    solver.run(3)
    return GpuBlueprintAgent(tree, solver.average_strategy_tables().astype(np.float64), sampler, subgame_search=False)


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = MultiStackBlueprintAgent({50.0: tiny_agent(50.0), 100.0: tiny_agent(100.0), 200.0: tiny_agent(200.0)})

    def test_effective_stack_selects_nearest_depth(self) -> None:
        cases = {40: 50.0, 60: 50.0, 90: 100.0, 130: 100.0, 175: 200.0, 250: 200.0}
        for chips_bb, expected_depth in cases.items():
            engine = HeadsUpHoldem(initial_stack=chips_bb * 20, small_blind=10, big_blind=20)
            engine.new_hand()
            chosen = self.router._route(engine, engine.current_player)
            self.assertEqual(
                chosen.tree.config.stack_bb, expected_depth, f"{chips_bb}bb routed to {chosen.tree.config.stack_bb}"
            )

    def test_route_locked_for_the_whole_hand(self) -> None:
        engine = HeadsUpHoldem(initial_stack=2000, small_blind=10, big_blind=20)  # 100bb
        engine.new_hand()
        first = self.router._route(engine, engine.current_player)
        # Even if stacks shrink mid-hand, the same sub-agent must serve it.
        engine.stacks[0] = 200  # would route to 50bb if recomputed
        again = self.router._route(engine, engine.current_player)
        self.assertIs(first, again)

    def test_plays_full_hands_without_illegal_actions(self) -> None:
        engine = HeadsUpHoldem(rng=random.Random(4), initial_stack=2000, small_blind=10, big_blind=20)
        for _ in range(15):
            while not engine.hand_complete:
                player = engine.current_player
                if player == 1:
                    self.router.execute(engine, 1, self.router.select(engine, 1))
                else:
                    legal = engine.legal_actions(0)
                    engine.act(0, "check" if legal.get("check") else "call" if legal.get("call") else "fold")
            self.router.observe_completed_hand(engine, 1)
            engine.new_hand()

    def test_depth_summary_and_search_toggle(self) -> None:
        self.assertEqual(set(self.router.depth_summary()), {50.0, 100.0, 200.0})
        self.router.subgame_search = True
        self.assertTrue(self.router.subgame_search)
        self.router.subgame_search = False
        self.assertFalse(self.router.subgame_search)


if __name__ == "__main__":
    unittest.main()
