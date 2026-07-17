"""Delta tracking and multiprocess blueprint training."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.solver.mccfr import LinearMCCFR, StrategyTable


class DeltaTrackingTests(unittest.TestCase):
    def test_collect_and_apply_round_trip(self) -> None:
        source = StrategyTable(3)
        source._regret_row("a")[:] = [1.0, 2.0, 3.0]
        source._strategy_row("a")[:] = [0.5, 0.5, 0.0]

        source.begin_delta()
        source._regret_row("a")[0] += 4.0
        source._regret_row("b")[1] += 7.0
        delta = source.collect_delta()

        self.assertEqual(set(delta), {"a", "b"})
        np.testing.assert_allclose(delta["a"][0], [4.0, 0.0, 0.0])
        np.testing.assert_allclose(delta["b"][0], [0.0, 7.0, 0.0])

        target = StrategyTable(3)
        target._regret_row("a")[:] = [1.0, 2.0, 3.0]
        target.apply_delta(delta)
        np.testing.assert_allclose(target.regrets["a"], [5.0, 2.0, 3.0])
        np.testing.assert_allclose(target.regrets["b"], [0.0, 7.0, 0.0])

    def test_apply_delta_on_table_unpickled_without_delta_slots(self) -> None:
        # Checkpoints written before delta tracking existed unpickle without
        # the _touched/_baseline slots; apply_delta must still work.
        table = StrategyTable(2)
        table._regret_row("a")[0] = 1.0
        restored = pickle.loads(pickle.dumps(table))
        for slot in ("_touched", "_baseline_regrets", "_baseline_sums"):
            try:
                delattr(restored, slot)
            except AttributeError:
                pass
        restored.apply_delta({"a": (np.array([2.0, 0.0]), np.array([0.0, 0.0]))})
        np.testing.assert_allclose(restored.regrets["a"], [3.0, 0.0])

    def test_tracking_disabled_by_default(self) -> None:
        table = StrategyTable(2)
        table._regret_row("x")[0] = 1.0  # must not require begin_delta
        self.assertIsNone(table._touched)

    def test_untouched_keys_are_not_shipped(self) -> None:
        table = StrategyTable(2)
        table._regret_row("seen")[0] = 1.0
        table.begin_delta()
        table._regret_row("seen")  # touched but unchanged
        delta = table.collect_delta()
        self.assertEqual(delta, {})

    def test_solver_convergence_survives_delta_mode(self) -> None:
        from backend.solver import exploitability
        from backend.solver.games import KuhnPoker

        game = KuhnPoker()
        solver = LinearMCCFR(game, seed=3)
        solver.table.begin_delta()
        for _ in range(40):
            solver.run(500)
            solver.table.collect_delta()
        self.assertLess(exploitability(game, solver.average_policy), 0.01)


class ParallelTrainingSmokeTests(unittest.TestCase):
    def test_two_workers_train_and_checkpoint(self) -> None:
        from backend.abstraction.buckets import AbstractionConfig, CardAbstraction
        from backend.solver import blueprint as bp
        from backend.solver.parallel import train_parallel

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = AbstractionConfig(
                flop_buckets=8,
                turn_buckets=8,
                river_buckets=4,
                fit_samples_per_street=80,
                flop_scenarios=8,
                opponents_per_scenario=6,
                seed=9,
            )
            abstraction = CardAbstraction(config=config).fit()
            abstraction_path = base / "abstraction.npz"
            abstraction.save(abstraction_path)

            with (
                patch.object(bp, "ABSTRACTION_PATH", abstraction_path),
                patch.object(bp, "BLUEPRINT_PATH", base / "blueprint.pkl"),
                patch.object(bp, "TELEMETRY_PATH", base / "telemetry.json"),
                patch.object(bp, "DATA_DIR", base),
            ):
                train_parallel(iterations=20, workers=2, chunk=5, save_every=10, seed=1, progress=False)

                self.assertTrue((base / "blueprint.pkl").exists())
                with open(base / "blueprint.pkl", "rb") as handle:
                    payload = pickle.load(handle)
                self.assertEqual(payload["iteration"], 20)
                self.assertGreater(len(payload["table"]), 20)


if __name__ == "__main__":
    unittest.main()
