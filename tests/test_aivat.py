"""AIVAT chance control variates (backend/eval/aivat.py).

Two properties on locally simulated hands (two scripted styles, so hands are
luck-driven and cheap):
1. UNBIASED: corrections average ~0, so corrected mean ~= raw mean.
2. USEFUL: corrected per-hand results have materially lower variance.
"""

import random
import statistics
import unittest

from backend.eval.aivat import ChanceCorrector
from backend.poker import HeadsUpHoldem
from backend.styles import style_action
from backend.rl_env import execute_action


def play_hand(seed: int):
    engine = HeadsUpHoldem(initial_stack=2000, small_blind=10, big_blind=20, rng=random.Random(seed))
    corrector = ChanceCorrector(engine, seat=0, samples=8, alternatives=12, seed=seed)
    before = engine.stacks[0]
    safety = 0
    while not engine.hand_complete and safety < 200:
        player = engine.current_player
        style = "calling_station" if player == 0 else "loose_aggressive"
        choice = style_action(engine, player, style)
        execute_action(engine, player, choice)
        corrector.observe(engine)
        safety += 1
    raw = (engine.stacks[0] - before) / engine.big_blind
    return raw, raw - corrector.total_bb(engine.big_blind)


class AivatTests(unittest.TestCase):
    def test_unbiased_and_variance_reducing(self) -> None:
        raw, corrected = [], []
        for seed in range(400):
            r, c = play_hand(seed)
            raw.append(r)
            corrected.append(c)
        raw_mean, cor_mean = statistics.fmean(raw), statistics.fmean(corrected)
        raw_std, cor_std = statistics.stdev(raw), statistics.stdev(corrected)
        # Unbiased: means agree within a small fraction of the per-hand std.
        self.assertLess(abs(raw_mean - cor_mean), 0.35 * raw_std / (len(raw) ** 0.5) * 4)
        # Variance reduction: chance-only variates measured ~16% variance cut
        # (std 86.8 -> 79.4) on this fixture — the rest of the variance lives
        # in hidden opponent cards / betting, which need the decision-variates
        # half of AIVAT (future). Guard against regressions below ~10%.
        self.assertLess(cor_std, 0.95 * raw_std, f"no variance reduction: {raw_std:.2f} -> {cor_std:.2f}")


if __name__ == "__main__":
    unittest.main()
