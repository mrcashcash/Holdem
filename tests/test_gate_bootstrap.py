"""First native depth may be compared with the currently deployed fallback."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.eval.gate import evaluate_gate
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import GpuActionConfig


def fake_agent(depth: float):
    return SimpleNamespace(
        tree=SimpleNamespace(config=GpuActionConfig(stack_bb=depth)),
        sampler=DealSampler(flop_samples=2, turn_samples=2),
        iteration=10,
        subgame_search=False,
    )


class BootstrapGateTests(unittest.TestCase):
    def test_cross_depth_incumbent_requires_explicit_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            challenger = root / "checkpoint.npz"
            incumbent = root / "fallback.npz"
            challenger.write_bytes(b"challenger")
            incumbent.write_bytes(b"incumbent")
            agents = {challenger.resolve(): fake_agent(20.0), incumbent.resolve(): fake_agent(100.0)}
            with patch("backend.eval.gate._load_agent", side_effect=lambda path: agents[path.resolve()]):
                with self.assertRaisesRegex(ValueError, "incumbent was trained for 100bb"):
                    evaluate_gate(
                        root,
                        20.0,
                        challenger_path=challenger,
                        incumbent_path=incumbent,
                        screen_pairs=1,
                        confirm_pairs=1,
                        retained_pairs=1,
                        lbr_pairs=0,
                    )

    def test_bootstrap_records_both_training_depths(self) -> None:
        duel = {
            "mean_bb_per_100": 1.0,
            "ci_low_bb_per_100": 0.1,
            "ci_high_bb_per_100": 1.9,
            "verdict": "PROMOTE",
            "hands": 2,
            "diagnostics": {
                "challenger": {"fallback_rate": 0.0, "mean_translation_gap_pot": 0.0},
                "champion": {"fallback_rate": 0.0, "mean_translation_gap_pot": 0.0},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            challenger = root / "checkpoint.npz"
            incumbent = root / "fallback.npz"
            challenger.write_bytes(b"challenger")
            incumbent.write_bytes(b"incumbent")
            agents = {challenger.resolve(): fake_agent(20.0), incumbent.resolve(): fake_agent(100.0)}
            with (
                patch("backend.eval.gate._load_agent", side_effect=lambda path: agents[path.resolve()]),
                patch("backend.eval.gate.head_to_head", return_value=duel),
            ):
                report = evaluate_gate(
                    root,
                    20.0,
                    challenger_path=challenger,
                    incumbent_path=incumbent,
                    screen_pairs=1,
                    confirm_pairs=1,
                    retained_pairs=1,
                    lbr_pairs=0,
                    allow_bootstrap_incumbent=True,
                    seed=7,
                )
        self.assertEqual(
            report["depth_comparison"],
            {
                "evaluation_depth_bb": 20.0,
                "challenger_trained_depth_bb": 20.0,
                "incumbent_trained_depth_bb": 100.0,
                "bootstrap_incumbent": True,
            },
        )
        self.assertTrue(report["promotion"]["eligible"])


if __name__ == "__main__":
    unittest.main()
