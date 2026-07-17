"""Phase 4 acceptance: the blueprint agent plays full hands through the real engine."""

import random
import unittest

from backend.abstraction.actions import ActionAbstraction
from backend.abstraction.buckets import AbstractionConfig, CardAbstraction
from backend.agents.blueprint_agent import BlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.solver.holdem import AbstractHoldem
from backend.solver.mccfr import LinearMCCFR


class BlueprintAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=5,
            fit_samples_per_street=120,
            flop_scenarios=12,
            opponents_per_scenario=8,
            seed=6,
        )
        abstraction = CardAbstraction(config=config).fit()
        game = AbstractHoldem(abstraction, ActionAbstraction(), stack_bb=50.0)
        solver = LinearMCCFR(game, seed=1)
        solver.run(100)
        cls.agent = BlueprintAgent(game, solver.table)

    def test_plays_many_full_hands_without_illegal_actions(self) -> None:
        engine = HeadsUpHoldem(rng=random.Random(11))
        hands = 0
        while hands < 40:
            while not engine.hand_complete:
                player = engine.current_player
                if player == 1:
                    choice = self.agent.select(engine, 1)
                    self.agent.execute(engine, 1, choice)  # raises InvalidAction on a contract breach
                else:
                    legal = engine.legal_actions(0)
                    if legal.get("check"):
                        engine.act(0, "check")
                    elif legal.get("call"):
                        engine.act(0, "call")
                    else:
                        engine.act(0, "fold")
            self.agent.observe_completed_hand(engine, 1)
            hands += 1
            engine.new_hand()
        self.assertEqual(hands, 40)

    def test_agent_folds_calls_and_raises_over_a_session(self) -> None:
        engine = HeadsUpHoldem(rng=random.Random(13))
        seen = set()
        for _ in range(200):
            while not engine.hand_complete:
                player = engine.current_player
                if player == 1:
                    choice = self.agent.select(engine, 1)
                    seen.add(choice)
                    self.agent.execute(engine, 1, choice)
                else:
                    legal = engine.legal_actions(0)
                    engine.act(0, "check" if legal.get("check") else "call" if legal.get("call") else "fold")
            engine.new_hand()
        # A sampled average strategy must mix: at minimum call and one of fold/raise/all-in.
        self.assertIn(1, seen)
        self.assertGreaterEqual(len(seen), 2)

    def test_try_load_returns_none_without_artifacts(self) -> None:
        from pathlib import Path

        missing = Path("does-not-exist-blueprint.pkl")
        self.assertIsNone(BlueprintAgent.try_load(missing, missing))

    def test_select_survives_opponent_off_tree_raises(self) -> None:
        engine = HeadsUpHoldem(rng=random.Random(17))
        # Human makes an awkward off-tree raise; agent must still act legally.
        while not engine.hand_complete:
            player = engine.current_player
            if player == 1:
                choice = self.agent.select(engine, 1)
                self.agent.execute(engine, 1, choice)
            else:
                legal = engine.legal_actions(0)
                if legal.get("raise"):
                    minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
                    awkward = min(maximum, max(minimum, int(minimum * 1.37) + 7))
                    engine.act(0, "raise", awkward)
                elif legal.get("call"):
                    engine.act(0, "call")
                else:
                    engine.act(0, "check")


if __name__ == "__main__":
    unittest.main()
