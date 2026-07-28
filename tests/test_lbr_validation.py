"""LBR validation: the probe must find exploits whose size is known in advance.

`docs/STATUS.md` §4 lists three eval harnesses that produced believable wrong
numbers. LBR is the instrument that is supposed to tell us how far the agent is
from equilibrium, so it gets the same treatment: before its number about a real
blueprint is believed, it must reproduce a known result against opponents whose
exploitability is not in doubt.

The anchor is exact. Against an opponent that folds whenever folding is legal
(and checks when it is free), a best-responding probe simply takes the pot every
hand:

* probe on the button posts 50, raises, the opponent folds -> probe nets +100
* probe in the big blind posts 100, the opponent folds its button -> nets +50

Averaged over the duplicate seat-swapped pair that is exactly +0.75 bb/hand =
**+75 bb/100, with zero variance**. Any other reading is a probe or accounting
defect. (The number coincides with the retracted +75 bb/100 blind-inflation
artifact; here it is the correct answer, arrived at for a different reason.)

The ordering test is the second half: a calling station must measure far more
exploitable than the trained champion. A probe that cannot separate those two is
useless as a guard, however plausible its absolute number looks.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from backend.solver.gpu.tree import CHECK_CALL, FOLD

CHECKPOINTS = (
    Path("backend/data/gpu_blueprint_200bb/champion.npz"),
    Path("backend/data/gpu_blueprint/champion.npz"),
)


class ConstantStrategy:
    """Strategy table that always plays `preferred` wherever it is legal.

    Substituted for a real agent's table so that every LBR code path — node
    location, action translation, the range posterior and the fold-response
    query — runs completely unmodified against an opponent whose policy is
    known. Supports the two indexing forms the probe uses:
    `strategy[node, bucket]` and `strategy[node, bucket_array, action]`.
    """

    def __init__(self, tree, preferred: int) -> None:
        self.tree = tree
        self.preferred = preferred
        rows = np.zeros((len(tree), tree.config.num_actions), dtype=np.float64)
        for node in range(len(tree)):
            legal = np.asarray(tree.legal[node], dtype=bool)
            if not legal.any():
                continue
            if legal[preferred]:
                choice = preferred
            elif legal[CHECK_CALL]:
                # Folding is illegal exactly when checking is free, so a
                # "always fold" opponent checks rather than picking at random.
                choice = CHECK_CALL
            else:  # pragma: no cover - one of the above always holds
                choice = int(np.argmax(legal))
            rows[node, choice] = 1.0
        self.rows = rows
        self.size = rows.size

    def __getitem__(self, key):
        node, bucket = key[0], key[1]
        row = self.rows[node]  # deliberately bucket-independent
        if len(key) > 2:
            value = row[key[2]]
            if np.ndim(bucket):
                return np.full(np.shape(bucket), value, dtype=np.float64)
            return value
        return row


def _load_agent():
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    for path in CHECKPOINTS:
        if path.exists():
            agent = GpuBlueprintAgent.try_load(path)
            if agent is not None:
                return agent, path
    return None, None


class LbrValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        agent, path = _load_agent()
        if agent is None:
            raise unittest.SkipTest(f"no blueprint checkpoint found in {[str(p) for p in CHECKPOINTS]}")
        cls.agent = agent
        cls.path = path

    def _probe(self, preferred: int | None, pairs: int) -> dict:
        from backend.eval.lbr import local_best_response_probe

        agent, _ = _load_agent()
        if preferred is not None:
            agent.strategy = ConstantStrategy(agent.tree, preferred)
        return local_best_response_probe(
            agent,
            hands=pairs * 2,
            seed=7,
            stack_bb=agent.tree.config.stack_bb,
        )

    def test_always_fold_opponent_is_exploited_for_exactly_75_bb_per_100(self) -> None:
        report = self._probe(FOLD, pairs=25)
        self.assertAlmostEqual(report["lbr_bb_per_100"], 75.0, places=4, msg=report)
        # Zero variance: every duplicate pair yields the same 0.75 bb.
        self.assertAlmostEqual(report["ci_low_bb_per_100"], 75.0, places=4, msg=report)
        self.assertAlmostEqual(report["ci_high_bb_per_100"], 75.0, places=4, msg=report)

    def test_calling_station_is_far_more_exploitable_than_the_champion(self) -> None:
        station = self._probe(CHECK_CALL, pairs=25)
        champion = self._probe(None, pairs=25)
        self.assertGreater(station["lbr_bb_per_100"], 100.0, station)
        self.assertGreater(
            station["lbr_bb_per_100"],
            champion["lbr_bb_per_100"],
            f"probe cannot separate a calling station from the champion: "
            f"station={station['lbr_bb_per_100']} champion={champion['lbr_bb_per_100']}",
        )

    def test_a_shove_bot_and_a_maniac_are_both_found_highly_exploitable(self) -> None:
        """The third and fourth broken references, in the aggressive direction.

        always-fold and the calling station are passive; a probe could in
        principle score them well by accident. A shove-bot (all-in always) and a
        maniac (smallest raise always) fail in the opposite direction, so a probe
        that ranks all of them correctly is discriminating rather than lucky.
        """
        from backend.solver.gpu.tree import ALL_IN

        # Raise ids start at 3; the first one is the smallest configured size.
        smallest_raise = 3
        champion = self._probe(None, pairs=25)
        for label, preferred in (("shove-bot", ALL_IN), ("maniac", smallest_raise)):
            report = self._probe(preferred, pairs=25)
            self.assertGreater(
                report["lbr_bb_per_100"],
                50.0,
                f"{label} should be grossly exploitable: {report['lbr_bb_per_100']}",
            )
            self.assertGreater(
                report["lbr_bb_per_100"],
                champion["lbr_bb_per_100"],
                f"probe ranks {label} as no worse than the champion: "
                f"{report['lbr_bb_per_100']} vs {champion['lbr_bb_per_100']}",
            )

    def test_probe_menu_offers_multiple_sizes(self) -> None:
        report = self._probe(CHECK_CALL, pairs=6)
        self.assertGreaterEqual(
            report["diagnostics"].get("probe_sizes", 0),
            4,
            f"LBR is probing too few bet sizes to bound exploitability: {report['diagnostics']}",
        )


if __name__ == "__main__":
    unittest.main()
