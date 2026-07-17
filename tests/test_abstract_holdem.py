"""Mechanics tests for the abstracted HUNL game (Phase 3 acceptance)."""

import random
import unittest

from backend.abstraction.actions import ALL_IN, CHECK_CALL, FOLD, ActionAbstraction
from backend.abstraction.buckets import AbstractionConfig, CardAbstraction
from backend.solver.holdem import AbstractHoldem
from backend.solver.mccfr import LinearMCCFR


def tiny_game() -> AbstractHoldem:
    config = AbstractionConfig(
        flop_buckets=10,
        turn_buckets=10,
        river_buckets=5,
        fit_samples_per_street=120,
        flop_scenarios=12,
        opponents_per_scenario=8,
        seed=4,
    )
    abstraction = CardAbstraction(config=config).fit()
    return AbstractHoldem(abstraction, ActionAbstraction(), stack_bb=50.0)


class AbstractHoldemMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.game = tiny_game()

    def _play_random_hand(self, rng: random.Random):
        state = self.game.initial_state()
        decisions = 0
        while not state.is_terminal():
            self.assertLess(decisions, 200, "hand did not terminate")
            if state.is_chance():
                state = state.sample_chance(rng)
                continue
            actions = state.legal_actions()
            self.assertTrue(actions, "decision node with no legal actions")
            state = state.child(rng.choice(list(actions)))
            decisions += 1
        return state

    def test_random_playouts_terminate_and_are_zero_sum(self) -> None:
        rng = random.Random(0)
        for _ in range(300):
            terminal = self._play_random_hand(rng)
            self.assertAlmostEqual(terminal.utility(0) + terminal.utility(1), 0.0, places=9)

    def test_utility_bounded_by_stack(self) -> None:
        rng = random.Random(1)
        for _ in range(300):
            terminal = self._play_random_hand(rng)
            self.assertLessEqual(abs(terminal.utility(0)), 50.0 + 1e-9)

    def test_blinds_and_first_action(self) -> None:
        state = self.game.initial_state()
        self.assertEqual(state.committed, (0.5, 1.0))
        rng = random.Random(2)
        state = state.sample_chance(rng)
        self.assertEqual(state.current_player(), 0)  # button acts first preflop
        self.assertIn(FOLD, state.legal_actions())

    def test_button_fold_pays_small_blind(self) -> None:
        rng = random.Random(3)
        state = self.game.initial_state().sample_chance(rng)
        terminal = state.child(FOLD)
        self.assertTrue(terminal.is_terminal())
        self.assertEqual(terminal.utility(0), -0.5)
        self.assertEqual(terminal.utility(1), 0.5)

    def test_limp_gives_big_blind_the_option(self) -> None:
        rng = random.Random(4)
        state = self.game.initial_state().sample_chance(rng)
        state = state.child(CHECK_CALL)  # button limps
        self.assertFalse(state.is_chance())
        self.assertEqual(state.current_player(), 1)  # big blind still to act
        state = state.child(CHECK_CALL)  # big blind checks the option
        self.assertTrue(state.is_chance())  # flop deal

    def test_postflop_big_blind_acts_first(self) -> None:
        rng = random.Random(5)
        state = self.game.initial_state().sample_chance(rng)
        state = state.child(CHECK_CALL).child(CHECK_CALL).sample_chance(rng)
        self.assertEqual(state.street, 1)
        self.assertEqual(len(state.board), 3)
        self.assertEqual(state.current_player(), 1)

    def test_all_in_call_runs_out_the_board(self) -> None:
        rng = random.Random(6)
        state = self.game.initial_state().sample_chance(rng)
        state = state.child(ALL_IN).child(CHECK_CALL)
        self.assertTrue(state.is_chance())
        state = state.sample_chance(rng)
        self.assertTrue(state.is_terminal())
        self.assertEqual(len(state.board), 5)
        # Both stacks are fully committed (blinds included): matched pot is 50 bb.
        self.assertIn(abs(state.utility(0)), (0.0, 50.0))

    def test_infoset_key_hides_opponent_cards(self) -> None:
        rng = random.Random(7)
        state = self.game.initial_state().sample_chance(rng)
        key = state.infoset_key()
        self.assertIsInstance(key, bytes)
        self.assertEqual(key[0], 0)  # street byte
        self.assertEqual(len(key), 3)  # street + 16-bit bucket, no history yet

    def test_mccfr_smoke_runs_on_the_full_game(self) -> None:
        solver = LinearMCCFR(self.game, seed=8)
        solver.run(50)
        self.assertGreater(len(solver.table), 50)


if __name__ == "__main__":
    unittest.main()
