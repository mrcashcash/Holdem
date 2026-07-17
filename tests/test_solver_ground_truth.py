"""Ground-truth validation of the Linear MCCFR solver (Phase 1 acceptance).

Kuhn poker has a known Nash value (-1/18 for player 0) and tiny size, so the
solver must drive exploitability near zero. Leduc is larger; we require the
exploitability curve to decrease materially with more iterations and to end
below a threshold consistent with published (MC)CFR curves.
"""

import unittest

from backend.solver import LinearMCCFR, best_response_value, exploitability
from backend.solver.games import KuhnPoker, LeducPoker


class KuhnGroundTruthTests(unittest.TestCase):
    def test_exploitability_converges_to_nash(self) -> None:
        game = KuhnPoker()
        solver = LinearMCCFR(game, seed=7)
        solver.run(20000)

        residual = exploitability(game, solver.average_policy)

        # Nash exploitability is 0; the ante is 1 chip, so 0.005 chips is
        # 0.5% of the ante — well inside "converged" for 20k iterations.
        self.assertLess(residual, 0.005)

    def test_game_value_matches_theory(self) -> None:
        game = KuhnPoker()
        solver = LinearMCCFR(game, seed=11)
        solver.run(20000)

        # Player 1's best response to a near-Nash player-0 strategy caps
        # player 0's value at the theoretical -1/18 = -0.0556.
        value_for_player_1 = best_response_value(game, solver.average_policy, 1)
        self.assertAlmostEqual(value_for_player_1, 1.0 / 18.0, delta=0.01)

    def test_terminal_payoffs_are_zero_sum(self) -> None:
        game = KuhnPoker()

        def walk(state) -> None:
            if state.is_terminal():
                self.assertAlmostEqual(state.utility(0) + state.utility(1), 0.0)
                return
            if state.is_chance():
                for successor, _ in state.chance_outcomes():
                    walk(successor)
                return
            for action in state.legal_actions():
                walk(state.child(action))

        walk(game.initial_state())


class LeducGroundTruthTests(unittest.TestCase):
    def test_exploitability_decreases_and_converges(self) -> None:
        game = LeducPoker()
        solver = LinearMCCFR(game, seed=3)

        solver.run(1000)
        early = exploitability(game, solver.average_policy)
        solver.run(24000)
        late = exploitability(game, solver.average_policy)

        self.assertLess(late, early * 0.5)
        # Measured curve: 0.56 @ 1k -> 0.11 @ 25k -> 0.07 @ 50k iterations.
        self.assertLess(late, 0.15)

    def test_terminal_payoffs_are_zero_sum_on_sampled_playouts(self) -> None:
        import random

        game = LeducPoker()
        rng = random.Random(5)
        for _ in range(500):
            state = game.initial_state()
            while not state.is_terminal():
                if state.is_chance():
                    state = state.sample_chance(rng)
                else:
                    state = state.child(rng.choice(list(state.legal_actions())))
            self.assertAlmostEqual(state.utility(0) + state.utility(1), 0.0)

    def test_pruning_does_not_break_convergence(self) -> None:
        game = KuhnPoker()
        solver = LinearMCCFR(game, seed=13, pruning_threshold=-25.0, pruning_warmup_iterations=2000)
        solver.run(20000)

        self.assertLess(exploitability(game, solver.average_policy), 0.01)


if __name__ == "__main__":
    unittest.main()
