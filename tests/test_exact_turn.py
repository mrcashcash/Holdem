"""Exact-card turn sampler (P1.2).

The blueprint stores turn strategy in 150 buckets and river strategy in 30. That
is the measured ceiling: LBR beats the serving champion by ~291 bb/100. Re-solving
at the blueprint's own resolution cannot help by construction, which is why the
2026-07-23 bucketed turn search measured a regression rather than a wash.

`ExactTurnSampler` removes the card abstraction from both streets: every private
combo is its own bucket. These tests pin the properties the solver depends on --
in particular that a combo's turn bucket is the SAME index in every river runout,
because that is what lets turn regrets accumulate per combo instead of smearing
across runouts.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from backend.search.exact_turn import TURN_FRACTIONS, TURN_RAISE_CAP, ExactTurnSampler
from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, combos, score_all_combos

BOARD = (0, 17, 30, 43)


class ExactTurnSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sampler = ExactTurnSampler(BOARD)

    def test_rejects_malformed_boards(self) -> None:
        with self.assertRaises(ValueError):
            ExactTurnSampler((0, 1, 2))  # three cards
        with self.assertRaises(ValueError):
            ExactTurnSampler((0, 1, 2, 3, 4))  # five cards
        with self.assertRaises(ValueError):
            ExactTurnSampler((0, 1, 2, 2))  # duplicate

    def test_bucket_counts_are_identity_on_turn_and_river(self) -> None:
        self.assertEqual(self.sampler.bucket_counts(), (1, 1, NUM_COMBOS, NUM_COMBOS))

    def test_enumerates_exactly_the_48_possible_rivers(self) -> None:
        self.assertEqual(len(self.sampler.rivers), 48)
        self.assertEqual(len(set(self.sampler.rivers)), 48)
        self.assertFalse(set(self.sampler.rivers) & set(BOARD))
        self.assertEqual(len(self.sampler.enumerate_deals()), 48)

    def test_buckets_are_the_combo_index_and_validity_matches_scoring(self) -> None:
        for river in self.sampler.rivers:
            deal = self.sampler.deal_for_river(river)
            scores = score_all_combos(BOARD + (river,))
            np.testing.assert_array_equal(deal.valid, scores >= 0)
            valid = deal.valid
            expected = np.arange(NUM_COMBOS)[valid]
            np.testing.assert_array_equal(deal.buckets[2][valid], expected)
            np.testing.assert_array_equal(deal.buckets[3][valid], expected)
            # Blocked combos must be -1 on every street, so the solver's
            # clamp-to-zero cannot smuggle them into bucket 0.
            self.assertTrue(np.all(deal.buckets[:, ~valid] == -1))

    def test_turn_bucket_is_river_independent(self) -> None:
        """The property that makes turn regrets accumulate per combo."""
        first, second = self.sampler.rivers[0], self.sampler.rivers[20]
        left = self.sampler.deal_for_river(first)
        right = self.sampler.deal_for_river(second)
        shared = left.valid & right.valid
        self.assertGreater(shared.sum(), 1000)
        np.testing.assert_array_equal(left.buckets[2][shared], right.buckets[2][shared])

    def test_board_and_river_cards_block_the_right_combos(self) -> None:
        board_blocked = np.zeros(NUM_COMBOS, dtype=bool)
        for card in BOARD:
            board_blocked |= CARD_IN_COMBO[card]
        for river in (self.sampler.rivers[0], self.sampler.rivers[11]):
            deal = self.sampler.deal_for_river(river)
            # Anything sharing a card with the board is dead in every runout.
            self.assertFalse(np.any(deal.valid & board_blocked))
            # Anything sharing the river card is dead in this runout only.
            self.assertFalse(np.any(deal.valid & CARD_IN_COMBO[river]))
            expected_valid = ~(board_blocked | CARD_IN_COMBO[river])
            np.testing.assert_array_equal(deal.valid, expected_valid)

    def test_valid_combo_count_is_the_combinatorial_answer(self) -> None:
        # 52 - 5 board/river cards = 47 live cards -> C(47,2) = 1081 combos.
        deal = self.sampler.deal_for_river(self.sampler.rivers[3])
        self.assertEqual(int(deal.valid.sum()), 47 * 46 // 2)

    def test_sample_returns_one_of_the_enumerated_deals(self) -> None:
        rng = random.Random(5)
        seen = set()
        for _ in range(400):
            deal = self.sampler.sample(rng)
            self.assertEqual(len(deal.board), 5)
            self.assertEqual(deal.board[:4], BOARD)
            seen.add(deal.board[4])
        # 400 uniform draws from 48 rivers should cover nearly all of them.
        self.assertGreater(len(seen), 40)
        self.assertTrue(seen <= set(self.sampler.rivers))

    def test_deals_are_cached_not_rebuilt(self) -> None:
        # Identity, not equality: rebuilding per sample would make the solver
        # pay board scoring on every iteration.
        river = self.sampler.rivers[7]
        self.assertIs(self.sampler.deal_for_river(river), self.sampler.deal_for_river(river))


class ExactTurnSolveTests(unittest.TestCase):
    """A real exact-card turn+river solve must run and produce a valid policy."""

    def test_solve_produces_a_normalized_exact_card_strategy(self) -> None:
        import torch

        from backend.solver.gpu.cfr import VectorCFR
        from backend.solver.gpu.tree import (
            DECISION,
            BettingRootState,
            BettingTree,
            GpuActionConfig,
        )

        config = GpuActionConfig(
            preflop_fractions=(1.0,),
            postflop_fractions=TURN_FRACTIONS,
            max_raises_per_street=TURN_RAISE_CAP,
            stack_bb=100.0,
        )
        root = BettingRootState(
            street=2, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
            stacks=(90.0, 90.0), acted=(False, False), raises=0, last_increment=1.0,
        )
        tree = BettingTree(config, root_state=root)
        sampler = ExactTurnSampler(BOARD)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        solver = VectorCFR(tree, sampler, device=device, seed=3, averaging_delay=2)

        # Compact storage must allocate exact-card rows for turn AND river.
        report = solver.storage_report()
        self.assertEqual(tuple(solver.bucket_counts), (1, 1, NUM_COMBOS, NUM_COMBOS))

        solver.run(6)
        strategy = solver.average_strategy_tables()
        self.assertEqual(strategy.shape[0], len(tree))
        self.assertTrue(np.all(np.isfinite(strategy)))

        combo_count = len(combos())
        checked = 0
        for node in range(len(tree)):
            if tree.kind[node] != DECISION:
                continue
            legal = np.asarray(tree.legal[node], dtype=bool)
            rows = strategy[node, :combo_count]
            self.assertTrue(np.all(rows[:, ~legal] == 0.0), f"node {node} used an illegal action")
            totals = rows.sum(axis=1)
            # Every combo row is a distribution (uniform where untouched).
            np.testing.assert_allclose(totals, 1.0, atol=1e-5)
            checked += 1
        self.assertGreater(checked, 0)
        self.assertGreater(report["total_rows"], 0)


class DrawFoldLeakTests(unittest.TestCase):
    """Exact cards must fix the documented draw-fold leak class.

    `docs/RESEARCH_ROADMAP.md` records a concrete exhibit (decision log, hand
    #222): 7s5s on 4s 5c 6s Kc -- pair + OESD + flush draw, ~19 outs, ~43% vs
    top pair -- folded **83%** to a 0.7-pot turn bet needing 29%. The scalar
    bucket merged that combo-draw with static ~45%-equity hands that correctly
    fold to polarized aggression, so the draw inherited their fold. This is the
    project's established decision-level evidence standard: the histogram
    abstraction was validated the same way.

    The assertion cuts both ways on purpose. A solver that simply calls
    everything is not better, only looser, so genuinely weak hands must still
    fold. Ranges here are uniform over live combos rather than blueprint-tracked,
    so this is a mechanism test, not a strength measurement -- but a 19-out draw
    should continue against any reasonable range.
    """

    SUIT = {"s": 0, "h": 1, "d": 2, "c": 3}
    RANK = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
            "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

    @classmethod
    def _cid(cls, text: str) -> int:
        return (cls.RANK[text[0]] - 2) * 4 + cls.SUIT[text[1]]

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from backend.solver.gpu.cfr import VectorCFR
        from backend.solver.gpu.tree import (
            DECISION,
            BettingRootState,
            BettingTree,
            GpuActionConfig,
        )

        cls.board = tuple(cls._cid(text) for text in ("4s", "5c", "6s", "Kc"))
        cls.bet_fraction = 0.7
        config = GpuActionConfig(
            preflop_fractions=(1.0,),
            postflop_fractions=(0.5, cls.bet_fraction, 1.0),
            max_raises_per_street=2,
            stack_bb=100.0,
        )
        root = BettingRootState(
            street=2, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
            stacks=(90.0, 90.0), acted=(False, False), raises=0, last_increment=1.0,
        )
        tree = BettingTree(config, root_state=root)
        bet_action = 3 + config.postflop_fractions.index(cls.bet_fraction)
        facing = int(tree.children[tree.root][bet_action])
        assert tree.kind[facing] == DECISION and int(tree.actor[facing]) == 0
        cls.tree = tree
        cls.facing = facing

        sampler = ExactTurnSampler(cls.board)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        solver = VectorCFR(tree, sampler, device=device, seed=17, averaging_delay=40)

        holdings = combos()
        live = np.ones(NUM_COMBOS, dtype=bool)
        for card in cls.board:
            live &= (holdings[:, 0] != card) & (holdings[:, 1] != card)
        cls.live = live
        ranges = np.zeros((2, NUM_COMBOS), dtype=np.float32)
        ranges[0] = live / live.sum()
        ranges[1] = live / live.sum()
        solver.root_reach = torch.as_tensor(ranges, device=solver.device)

        iterations = 400
        if device == "cuda":
            from backend.solver.gpu.graph import GraphRunner

            GraphRunner(solver, warmup=2).run(iterations, random.Random(5))
            torch.cuda.synchronize()
        else:  # pragma: no cover - CI fallback
            solver.run(iterations)
        cls.strategy = solver.average_strategy_tables()
        cls.index = {(int(a), int(b)): i for i, (a, b) in enumerate(holdings)}

    def _fold_probability(self, first: str, second: str) -> float:
        from backend.solver.gpu.tree import FOLD

        key = tuple(sorted((self._cid(first), self._cid(second))))
        row = self.strategy[self.facing, self.index[key]]
        return float(row[FOLD])

    def test_the_documented_hand_no_longer_folds(self) -> None:
        fold = self._fold_probability("7s", "5s")
        self.assertLess(
            fold, 0.25,
            f"7s5s still folds {fold:.3f} to a 0.7-pot turn bet "
            "(the 150-bucket blueprint folded 0.83)",
        )

    def test_big_draws_continue(self) -> None:
        for first, second in (("8s", "7s"), ("As", "2s"), ("8h", "7h")):
            fold = self._fold_probability(first, second)
            self.assertLess(fold, 0.35, f"{first}{second} folds {fold:.3f}")

    def test_trash_still_folds_so_this_is_discrimination_not_looseness(self) -> None:
        folds = [self._fold_probability(a, b) for a, b in (("Qd", "2h"), ("Jh", "3d"), ("9d", "2d"))]
        for fold in folds:
            self.assertGreater(fold, 0.75, f"weak hand folds only {fold:.3f}")
        draws = [self._fold_probability(a, b) for a, b in (("7s", "5s"), ("8s", "7s"), ("8h", "7h"))]
        self.assertGreater(
            float(np.mean(folds)) - float(np.mean(draws)), 0.5,
            "exact-card solve does not separate draws from trash",
        )


if __name__ == "__main__":
    unittest.main()
