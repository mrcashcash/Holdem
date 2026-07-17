"""The GPU blueprint agent plays legal hands through the real engine."""

import random
import unittest

import numpy as np

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig


class GpuAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = GpuActionConfig(
            preflop_fractions=(1.0,),
            postflop_fractions=(0.75,),
            max_raises_per_street=2,
            stack_bb=50.0,
        )
        tree = BettingTree(config)
        sampler = DealSampler(flop_samples=2, turn_samples=2)
        solver = VectorCFR(tree, sampler, device="cpu", seed=4)
        solver.run(8)
        cls.agent = GpuBlueprintAgent(tree, solver.average_strategy_tables().astype(np.float64), sampler)

    def test_plays_full_hands_without_illegal_actions(self) -> None:
        engine = HeadsUpHoldem(rng=random.Random(31))
        hands = 0
        while hands < 30:
            while not engine.hand_complete:
                player = engine.current_player
                if player == 1:
                    choice = self.agent.select(engine, 1)
                    self.agent.execute(engine, 1, choice)  # raises InvalidAction on breach
                else:
                    legal = engine.legal_actions(0)
                    if legal.get("raise") and hands % 3 == 0:
                        minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
                        engine.act(0, "raise", min(maximum, max(minimum, int(minimum * 1.61))))
                    elif legal.get("check"):
                        engine.act(0, "check")
                    elif legal.get("call"):
                        engine.act(0, "call")
                    else:
                        engine.act(0, "fold")
            hands += 1
            engine.new_hand()
        self.assertEqual(hands, 30)

    def test_try_load_returns_none_without_artifacts(self) -> None:
        from pathlib import Path

        self.assertIsNone(GpuBlueprintAgent.try_load(Path("missing-checkpoint.npz")))


if __name__ == "__main__":
    unittest.main()
