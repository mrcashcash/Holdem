"""Phase 5 acceptance: river subgame re-solving."""

import random
import unittest

from backend.abstraction.actions import ALL_IN, CHECK_CALL, FOLD, ActionAbstraction
from backend.abstraction.buckets import AbstractionConfig, CardAbstraction
from backend.agents.blueprint_agent import BlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.search.river import RiverSubgame, solve_river
from backend.solver.holdem import AbstractHoldem
from backend.solver.mccfr import LinearMCCFR


def card(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit


class RiverSubgameTests(unittest.TestCase):
    def test_random_playouts_are_zero_sum_and_terminate(self) -> None:
        board = (card(12, 0), card(11, 0), card(10, 0), card(3, 1), card(7, 2))
        blocked = set(board)
        combos = [
            (first, second)
            for first in range(52)
            if first not in blocked
            for second in range(first + 1, 52)
            if second not in blocked
        ]
        uniform = {combo: 1.0 for combo in combos}
        subgame = RiverSubgame(board, pot_start=10.0, stacks=(45.0, 45.0), ranges=(uniform, uniform))
        rng = random.Random(3)
        for _ in range(200):
            state = subgame.initial_state()
            steps = 0
            while not state.is_terminal():
                self.assertLess(steps, 60)
                if state.is_chance():
                    state = state.sample_chance(rng)
                else:
                    state = state.child(rng.choice(list(state.legal_actions())))
                steps += 1
            self.assertAlmostEqual(state.utility(0) + state.utility(1), 0.0, places=9)

    def test_solved_nuts_never_folds_to_a_bet(self) -> None:
        # Concentrated ranges (as produced by real range tracking) so the
        # solve visits every line densely: hero is polarized nuts/air,
        # villain always holds a medium made hand.
        board = (card(12, 0), card(11, 0), card(10, 0), card(3, 1), card(7, 2))
        nuts = (card(14, 0), card(13, 0))  # royal flush
        air = (card(2, 1), card(4, 3))
        medium = (card(10, 1), card(9, 3))  # pair of tens with a straight blocker
        hero_range = {nuts: 0.5, air: 0.5}
        villain_range = {medium: 1.0}
        subgame = RiverSubgame(board, pot_start=10.0, stacks=(45.0, 45.0), ranges=(hero_range, villain_range))
        solver = solve_river(subgame, iterations=500, seed=5)

        # Seat 1 (first to act) bets; seat 0 holds the nuts facing the bet.
        from dataclasses import replace as dc_replace

        state = dc_replace(subgame.initial_state(), combos=(nuts, medium))
        bet_actions = [a for a in state.legal_actions() if a >= 3]
        state = state.child(bet_actions[0])
        self.assertEqual(state.current_player(), 0)
        actions = list(state.legal_actions())
        probabilities = solver.table.average_strategy(state.infoset_key(), actions)
        self.assertIn(FOLD, actions)
        fold_probability = probabilities[actions.index(FOLD)]
        self.assertLess(fold_probability, 0.1)


class RiverIntegratedAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=5,
            fit_samples_per_street=120,
            flop_scenarios=12,
            opponents_per_scenario=8,
            seed=8,
        )
        abstraction = CardAbstraction(config=config).fit()
        game = AbstractHoldem(abstraction, ActionAbstraction(), stack_bb=50.0)
        solver = LinearMCCFR(game, seed=2)
        solver.run(60)
        cls.agent = BlueprintAgent(game, solver.table, river_search=True, river_iterations=60)

    def test_agent_with_river_search_plays_legal_hands(self) -> None:
        engine = HeadsUpHoldem(rng=random.Random(21))
        rivers_reached = 0
        hands = 0
        while hands < 25:
            while not engine.hand_complete:
                player = engine.current_player
                if player == 1:
                    if engine.street == 3:
                        rivers_reached += 1
                    choice = self.agent.select(engine, 1)
                    self.agent.execute(engine, 1, choice)
                else:
                    legal = engine.legal_actions(0)
                    engine.act(0, "check" if legal.get("check") else "call" if legal.get("call") else "fold")
            engine.new_hand()
            hands += 1
        self.assertGreater(rivers_reached, 0, "test never exercised the river path")


if __name__ == "__main__":
    unittest.main()
