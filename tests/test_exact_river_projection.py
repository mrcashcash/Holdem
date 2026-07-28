"""Phase 4 blueprint-projection regression tests (P1.1).

The 3,000-pair Phase 4 confirmation lost 25 of 1,889 river resolves (1.32%,
above its 1% eligibility limit) and every one reported the same cause:
"blueprint projection reached an incompatible public state". The exact resolver
uses real stacks, a richer size menu and a live mid-street root, so it
legitimately contains decisions where the coarse blueprint tree has already
terminated, has the other player acting, or has no matching child. Treating that
as an error threw away exact-card resolving for the whole hand.

These tests pin the repair: divergence *detaches* the affected subtree onto the
serving agent's safe-default policy and records the mismatch, and the projection
never raises. The four adversarial branches named in the plan — shallow stack,
all-in, raise cap, off-tree size — are covered directly, because those are
exactly where the two topologies part company.
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.solver.gpu.tree import ALL_IN, CHECK_CALL, DECISION, FOLD, BettingRootState, BettingTree

CHECKPOINTS = (
    Path("backend/data/gpu_blueprint_200bb/champion.npz"),
    Path("backend/data/gpu_blueprint/champion.npz"),
)

# A fixed river board; suits chosen to avoid an all-same-suit texture.
BOARD = (0, 17, 30, 43, 8)


def _stub_tree(legal: list[bool], fractions: tuple[float, ...]):
    """Minimal stand-in exposing only what `_map_action` reads."""
    return SimpleNamespace(
        legal=[np.array(legal, dtype=bool)],
        street=[3],
        config=SimpleNamespace(num_actions=len(legal), fractions=lambda street: fractions),
    )


def _load_agent():
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    for path in CHECKPOINTS:
        if path.exists():
            agent = GpuBlueprintAgent.try_load(path)
            if agent is not None:
                return agent
    return None


class MapActionPrecedenceTests(unittest.TestCase):
    """`_map_action` must fall back exactly as serving's translation does."""

    def test_unavailable_check_call_becomes_the_smallest_raise(self) -> None:
        from backend.search.exact_river import _map_action

        # Serving maps an unavailable check/call (a no_limp tree has no
        # open-limp branch) onto the SMALLEST legal raise, not onto a fold.
        source = _stub_tree([True, True, True, True], (0.5,))
        target = _stub_tree([True, False, True, True, True], (0.5, 1.0))
        mapped = _map_action(source, 0, CHECK_CALL, target, 0)
        self.assertEqual(mapped, 3, "check/call should map to the smallest legal raise")

    def test_small_unavailable_raise_becomes_check_call_not_all_in(self) -> None:
        from backend.search.exact_river import _map_action

        # Serving only promotes an untranslatable raise to an all-in when the
        # observed size exceeded 1.5 pot; a 0.5-pot raise must check/call.
        source = _stub_tree([True, True, True, True], (0.5,))
        target = _stub_tree([True, True, True, False], (0.5,))
        self.assertEqual(_map_action(source, 0, 3, target, 0), CHECK_CALL)

    def test_large_unavailable_raise_becomes_all_in(self) -> None:
        from backend.search.exact_river import _map_action

        source = _stub_tree([True, True, True, True], (2.5,))
        target = _stub_tree([True, True, True, False], (0.5,))
        self.assertEqual(_map_action(source, 0, 3, target, 0), ALL_IN)

    def test_unavailable_all_in_becomes_check_call(self) -> None:
        from backend.search.exact_river import _map_action

        source = _stub_tree([True, True, True, True], (0.5,))
        target = _stub_tree([True, True, False, True], (0.5,))
        self.assertEqual(_map_action(source, 0, ALL_IN, target, 0), CHECK_CALL)

    def test_unavailable_fold_becomes_check_call(self) -> None:
        from backend.search.exact_river import _map_action

        source = _stub_tree([True, True, True, True], (0.5,))
        target = _stub_tree([False, True, True, True], (0.5,))
        self.assertEqual(_map_action(source, 0, FOLD, target, 0), CHECK_CALL)

    def test_sized_raise_maps_to_the_nearest_available_fraction(self) -> None:
        from backend.search.exact_river import _map_action

        source = _stub_tree([True, True, True, True], (1.0,))
        target = _stub_tree([True, True, True, True, True, True], (0.25, 0.9, 2.0))
        self.assertEqual(_map_action(source, 0, 3, target, 0), 4, "1.0 pot is nearest 0.9")


class NodePotTests(unittest.TestCase):
    """The logged pot/SPR must be real, not a silently-zero placeholder.

    `matched_pot` is populated only on SHOWDOWN/HORIZON nodes and a fold's amount
    lives in `fold_loser_committed`, so the obvious read returns 0.0 at every
    decision node. A diagnostic that always logs 0.0 looks like data.
    """

    def _tree(self, committed, street_commit, stacks, acted, raises, last_increment, to_act):
        from backend.search.exact_river import _config

        root = BettingRootState(
            street=3, to_act=to_act, committed=committed, street_commit=street_commit,
            stacks=stacks, acted=acted, raises=raises, last_increment=last_increment,
        )
        return BettingTree(_config(None, 200.0), root_state=root)

    def test_matched_pot_is_twice_the_level_commitment(self) -> None:
        from backend.search.exact_river import _node_matched_pot

        tree = self._tree((10.0, 10.0), (0.0, 0.0), (190.0, 190.0), (False, False), 0, 1.0, 1)
        self.assertEqual(_node_matched_pot(tree, tree.root), 20.0)

    def test_matched_pot_facing_a_bet_uses_the_lower_commitment(self) -> None:
        from backend.search.exact_river import _node_matched_pot

        # 20 vs 33 committed: the matched level is 20, so the pot is 40.
        tree = self._tree((20.0, 33.0), (0.0, 13.0), (80.0, 3.0), (False, True), 1, 13.0, 0)
        self.assertEqual(_node_matched_pot(tree, tree.root), 40.0)

    def test_every_decision_node_reports_a_positive_pot(self) -> None:
        from backend.search.exact_river import _node_matched_pot

        from backend.solver.gpu.tree import DECISION as DECISION_KIND

        for stacks in ((190.0, 190.0), (5.0, 5.0), (0.5, 0.5)):
            tree = self._tree((10.0, 10.0), (0.0, 0.0), stacks, (False, False), 0, 1.0, 1)
            offenders = [
                node
                for node in range(len(tree))
                if tree.kind[node] == DECISION_KIND and _node_matched_pot(tree, node) <= 0.0
            ]
            self.assertEqual(offenders, [], f"stacks={stacks} produced zero-pot decisions")


class ProjectionRobustnessTests(unittest.TestCase):
    """The projection must survive every topology divergence without aborting."""

    @classmethod
    def setUpClass(cls) -> None:
        agent = _load_agent()
        if agent is None:
            raise unittest.SkipTest(f"no checkpoint in {[str(p) for p in CHECKPOINTS]}")
        cls.agent = agent

        from backend.search.exact_river import ExactRiverSampler
        from backend.search.gpu_subgame import partial_board_buckets

        cls.valid = ExactRiverSampler(BOARD)._deal.valid
        cls.bucket_row = partial_board_buckets(BOARD, agent.sampler, seed=11)[3]
        cls.river_nodes = [
            node
            for node in range(len(agent.tree))
            if agent.tree.kind[node] == DECISION and int(agent.tree.street[node]) == 3
        ]
        if not cls.river_nodes:
            raise unittest.SkipTest("blueprint tree has no river decision nodes")

    def _cases(self) -> list[tuple[str, BettingRootState, dict | None]]:
        """Root states spanning the four divergence branches from the plan."""
        return [
            (
                "deep fresh river",
                BettingRootState(
                    street=3, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
                    stacks=(190.0, 190.0), acted=(False, False), raises=0, last_increment=1.0,
                ),
                None,
            ),
            (
                "shallow stack",
                BettingRootState(
                    street=3, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
                    stacks=(5.0, 5.0), acted=(False, False), raises=0, last_increment=1.0,
                ),
                None,
            ),
            (
                "near all-in (sub-blind stacks)",
                BettingRootState(
                    street=3, to_act=1, committed=(60.0, 60.0), street_commit=(0.0, 0.0),
                    stacks=(0.5, 0.5), acted=(False, False), raises=0, last_increment=1.0,
                ),
                None,
            ),
            (
                "raise cap already reached",
                BettingRootState(
                    street=3, to_act=0, committed=(40.0, 48.0), street_commit=(8.0, 16.0),
                    stacks=(60.0, 52.0), acted=(True, True), raises=2, last_increment=8.0,
                ),
                None,
            ),
            (
                "asymmetric stacks, mid-street facing a bet",
                BettingRootState(
                    street=3, to_act=0, committed=(20.0, 33.0), street_commit=(0.0, 13.0),
                    stacks=(80.0, 3.0), acted=(False, True), raises=1, last_increment=13.0,
                ),
                None,
            ),
            (
                "off-tree observed size",
                BettingRootState(
                    street=3, to_act=0, committed=(18.0, 25.4), street_commit=(0.0, 7.4),
                    stacks=(90.0, 82.6), acted=(False, True), raises=1, last_increment=7.4,
                ),
                {
                    "action": "raise", "action_index": 2, "amount": 7.4,
                    "pot_before": 36.0, "to_call_before": 0.0, "current_bet_before": 0.0,
                },
            ),
        ]

    def test_projection_never_aborts_and_always_yields_a_valid_policy(self) -> None:
        from backend.search.exact_river import _config, _project_blueprint

        rng = random.Random(3)
        roots = rng.sample(self.river_nodes, min(6, len(self.river_nodes)))
        checked = 0
        detached_total = 0
        decisions_total = 0

        for label, root_state, observed in self._cases():
            tree = BettingTree(_config(observed, 200.0), root_state=root_state)
            for blueprint_root in roots:
                # Must not raise for ANY combination, including ones where the
                # blueprint actor disagrees with the exact actor.
                baseline, diagnostics = _project_blueprint(
                    self.agent, tree, blueprint_root, self.bucket_row, self.valid
                )
                checked += 1
                detached_total += diagnostics["projection_detached_nodes"]
                decisions_total += diagnostics["projection_decision_nodes"]

                policy = baseline.numpy()
                self.assertEqual(policy.shape[0], len(tree))
                self.assertEqual(policy.shape[2], tree.config.num_actions)
                self.assertTrue(np.all(np.isfinite(policy)), f"{label}: non-finite baseline")

                for node in range(len(tree)):
                    if tree.kind[node] != DECISION:
                        continue
                    legal = np.asarray(tree.legal[node], dtype=bool)
                    rows = policy[node]
                    # Illegal actions and board-colliding combos carry no mass.
                    self.assertTrue(
                        np.all(rows[:, ~legal] == 0.0),
                        f"{label}: node {node} put mass on an illegal action",
                    )
                    self.assertTrue(
                        np.all(rows[~self.valid] == 0.0),
                        f"{label}: node {node} put mass on a blocked combo",
                    )
                    totals = rows[self.valid].sum(axis=1)
                    np.testing.assert_allclose(
                        totals, 1.0, atol=1e-5,
                        err_msg=f"{label}: node {node} is not a distribution",
                    )
                self.assertLessEqual(
                    diagnostics["projection_detached_nodes"],
                    diagnostics["projection_decision_nodes"],
                )
        self.assertGreater(checked, 20, "test did not exercise enough combinations")
        # The repair is only meaningful if these cases really do diverge; if
        # nothing ever detached, the test would be proving nothing.
        self.assertGreater(
            detached_total, 0,
            "no divergence occurred, so this suite is not testing the repair",
        )
        print(
            f"\nprojection coverage: {checked} (tree, blueprint-root) pairs; "
            f"{detached_total}/{decisions_total} decision nodes detached"
        )

    def test_detachment_is_reported_with_actionable_detail(self) -> None:
        from backend.search.exact_river import _config, _project_blueprint

        # Force divergence: a sub-blind stack leaves the exact tree with only an
        # all-in/fold shape while the blueprint still has a full menu.
        root_state = BettingRootState(
            street=3, to_act=1, committed=(60.0, 60.0), street_commit=(0.0, 0.0),
            stacks=(0.5, 0.5), acted=(False, False), raises=0, last_increment=1.0,
        )
        tree = BettingTree(_config(None, 200.0), root_state=root_state)
        found = None
        for blueprint_root in self.river_nodes[:40]:
            _baseline, diagnostics = _project_blueprint(
                self.agent, tree, blueprint_root, self.bucket_row, self.valid
            )
            if diagnostics["projection_detached_roots"]:
                found = diagnostics
                break
        if found is None:
            self.skipTest("no divergent blueprint root among the sampled nodes")
        self.assertTrue(found["projection_detach_reasons"])
        sample = found["projection_detach_samples"][0]
        # Requirement 4: the mismatch must be diagnosable from the log alone.
        for key in ("reason", "exact_node", "exact_actor", "exact_street", "exact_pot_bb", "exact_legal"):
            self.assertIn(key, sample)


if __name__ == "__main__":
    unittest.main()
