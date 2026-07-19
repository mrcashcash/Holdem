"""Raise sizes at the table must match the blueprint's trained sizes.

Regression for the min-click 3-bet bug (2026-07-19): the agent's intended
0.75x/1.5x-pot 3-bets were being re-scaled through rl_env's legacy preflop
caps, collapsing every 3-bet toward the minimum (e.g. raise-to 83 facing an
open to 50 at 10/20).
"""

import random
import unittest

import numpy as np

from backend.agents.gpu_blueprint_agent import NEURAL_RAISE, GpuBlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig


class ThreeBetSizingTests(unittest.TestCase):
    def test_three_bet_executes_at_trained_pot_fraction(self) -> None:
        config = GpuActionConfig(
            preflop_fractions=(0.75, 1.5), postflop_fractions=(0.5, 1.0),
            max_raises_per_street=3, stack_bb=100.0,
        )
        tree = BettingTree(config)
        sampler = DealSampler(flop_samples=2, turn_samples=2)
        solver = VectorCFR(tree, sampler, device="cpu", seed=13)
        solver.run(4)
        agent = GpuBlueprintAgent(
            tree, solver.average_strategy_tables().astype(np.float64), sampler, subgame_search=False
        )

        # Recreate the observed spot: 10/20 blinds, button opens to 50.
        engine = HeadsUpHoldem(rng=random.Random(3))
        button = engine.button
        engine.act(button, "raise", 50)
        agent_player = 1 - button

        # Force a raise decision and check the executed amount.
        for _ in range(60):
            engine_copy = HeadsUpHoldem(rng=random.Random(3))
            engine_copy.act(engine_copy.button, "raise", 50)
            choice = agent.select(engine_copy, agent_player)
            if choice == NEURAL_RAISE:
                agent.execute(engine_copy, agent_player, choice)
                raise_to = engine_copy.round_bets[agent_player]
                # pot after open = 70, to_call = 30: trained sizes are
                # 0.75x100 -> raise-to 125 or 1.5x100 -> raise-to 200.
                self.assertIn(raise_to, (125, 200), f"3-bet executed at {raise_to}")
                return
        self.skipTest("agent never raised in 60 samples (legal but unhelpful)")


if __name__ == "__main__":
    unittest.main()
