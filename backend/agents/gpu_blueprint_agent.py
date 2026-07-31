"""Serve the dense GPU blueprint through the live game's agent contract.

Mirrors the real hand onto the flattened betting tree: public actions are
translated onto the coarse GPU menu (pseudo-harmonic for raises), and at our
decisions the average strategy at (node, street bucket) is sampled. Street
buckets are computed exactly as in training: blocker-corrected sort equity on
the current board (cached per board+street).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np

from backend.abstraction.actions import pseudo_harmonic_weights
from backend.abstraction.cards import preflop_class
from backend.poker import HeadsUpHoldem
from backend.rl_env import execute_action
from backend.solver.gpu.deals import DealSampler, NUM_COMBOS, combos, equity_from_scores, score_all_combos
from backend.solver.gpu.storage import CompactStrategy, CompactTableLayout
from backend.solver.gpu.tree import ALL_IN, CHECK_CALL, DECISION, FOLD, STREET_END, BettingTree, GpuActionConfig
from backend.vectorized_engine import card_id

NEURAL_FOLD, NEURAL_CHECK_CALL, NEURAL_RAISE, NEURAL_ALL_IN = 0, 1, 2, 3

#: How far the real shove-to-pot ratio may exceed the abstract one before a
#: translated ALL-IN is resized (see `GpuBlueprintAgent._all_in_choice`).
ALL_IN_GEOMETRY_TOLERANCE = 1.5

#: Absolute bound on a jam, as a multiple of the matched pot, whenever a smaller
#: raise is legal. Translation-matching alone does not cover every case: at a
#: raise-capped node the abstraction's ONLY aggressive action is all-in, so the
#: trained jam is itself 5-8x pot and no amount of ratio-matching shrinks it.
#: Both the jam and any resized bet are off-tree at such a node, so the choice is
#: between heuristics -- and risking 199bb to win 8bb is the worse one. 6.0 still
#: allows a genuine short-stack jam (behind under 6x pot jams untouched) and the
#: overbet-jam sizings that appear in real solver output.
ALL_IN_MAX_POT_MULTIPLE = 6.0

_COMBO_INDEX = {(int(a), int(b)): index for index, (a, b) in enumerate(combos())}


class GpuBlueprintAgent:
    """Drop-in serving agent backed by the dense GPU blueprint tables."""

    def __init__(
        self,
        tree: BettingTree,
        strategy: np.ndarray,
        sampler: DealSampler,
        iteration: int = 0,
        subgame_search: bool = True,
        subgame_iterations: int = 120,
    ) -> None:
        self.tree = tree
        # Legacy checkpoints use a dense ndarray; compact-v2 checkpoints use
        # CompactStrategy with the same [node, bucket] lookup contract.
        self.strategy = strategy
        self.sampler = sampler
        self.iteration = iteration
        self.ready = True
        self.subgame_search = subgame_search
        self.subgame_iterations = subgame_iterations
        # Safe (max-margin gadget) re-solving is the default: plain re-solving
        # trusts blueprint ranges for the opponent and measured as exploitable
        # by off-blueprint opponents. HOLDEM_SAFE_SEARCH=0 selects the plain
        # solver (A/B only).
        self.safe_search = os.environ.get("HOLDEM_SAFE_SEARCH", "1") != "0"
        # Flop search via the v0 169-bucket CFV net was deleted on 2026-07-28
        # with the rest of that pipeline (it measured -65 bb/100). Flop
        # depth-limited solving returns with the P3 net stack.
        # Phase 4 exact-card river resolving is independent of the retired
        # bucketed turn/river search. It is deliberately opt-in until its
        # on/off duel clears the promotion gate.
        self.exact_river_search = os.environ.get("HOLDEM_PHASE4_RIVER", "0") == "1"
        self.exact_river_iterations = max(
            12,
            int(os.environ.get("HOLDEM_PHASE4_ITERS", "80")),
        )
        self.exact_river_budget_ms = max(
            1,
            int(os.environ.get("HOLDEM_PHASE4_BUDGET_MS", "6000")),
        )
        # P1 continual resolving: exact-card turn AND river, one session per hand
        # carrying ranges across the street boundary. Supersedes the river-only
        # Phase 4 path when enabled. Off by default until its gate clears.
        self.continual_search = os.environ.get("HOLDEM_CONTINUAL", "0") == "1"
        self.continual_iterations = max(
            12,
            int(os.environ.get("HOLDEM_CONTINUAL_ITERS", "80")),
        )
        self.continual_budget_ms = max(
            1,
            int(os.environ.get("HOLDEM_CONTINUAL_BUDGET_MS", "8000")),
        )
        # Which streets the exact resolver handles. River-only by default: a
        # turn resolve costs 12-38 s at the standard 5-size menu while a river
        # resolve costs 2.4-5.3 s, and the river is where the blueprint is
        # weakest (30 buckets vs 150 on the turn). See backend/agents/serving.py.
        self.continual_streets: tuple[int, ...] = (3,)
        self._continual_sessions: dict[tuple[int, int], object] = {}
        self.last_continual_search: dict | None = None
        self._raise_fraction: float | None = None
        self._raise_target: int | None = None
        self._rng = random.Random(97)
        self._equity_cache: dict[tuple, float] = {}
        self._subgame_cache: dict[tuple, object] = {}
        self._river_sessions: dict[tuple[int, int], object] = {}
        self.last_river_search: dict | None = None
        self._node_pot_cache: dict[int, float] = {}
        #: Resize translated ALL-INs (see `_all_in_size`). **OFF by default.** The
        #: guard is not free: measured -268.82 bb/100 at 200bb and -124.00 at
        #: 100bb against a min-raiser, whose lines are the only ones that trigger
        #: it, and roughly neutral (-3.46 / -13.33) against everything else. A
        #: station calls any jam, so against it a 199bb shove is correct
        #: exploitation rather than a bug. Nothing has measured the guard as
        #: helpful, and this project does not serve that. The measurement that
        #: would settle it -- LBR guard-on vs guard-off -- has not been taken.
        self.all_in_geometry_guard = False
        self.all_in_geometry_tolerance = ALL_IN_GEOMETRY_TOLERANCE
        self.all_in_max_pot_multiple = ALL_IN_MAX_POT_MULTIPLE
        #: Set whenever a translated ALL-IN was resized; read by the decision log
        #: so an unusual bet size is explainable after the fact.
        self.last_all_in_rescale: dict | None = None

    @classmethod
    def try_load(cls, checkpoint_path: Path | None = None) -> "GpuBlueprintAgent | None":
        from backend.solver.gpu import train as gpu_train

        if checkpoint_path is None:
            # Prefer the promoted champion (backend/eval/promote.py) — the
            # newest raw checkpoint may not have passed the quality gate.
            champion = gpu_train.DATA_DIR / "champion.npz"
            checkpoint_path = champion if champion.exists() else gpu_train.CHECKPOINT_PATH
        path = checkpoint_path
        if not path.exists():
            return None
        payload = np.load(path, allow_pickle=False)
        config = GpuActionConfig(**json.loads(str(payload["config"])))
        sampler_state = json.loads(str(payload["sampler"]))
        # from_state restores fitted std edges; older checkpoints stored only
        # constructor kwargs (no std_edges key) — from_state handles both.
        sampler = DealSampler.from_state(sampler_state)
        tree = BettingTree(config)
        sums = payload["strategy_sums"]
        layout = CompactTableLayout(tree, sampler.bucket_counts())
        if sums.ndim == 3:
            # Legacy checkpoints stored a dense table for terminal nodes and
            # unused street buckets.  Serving used to normalize and retain that
            # entire 0.5-1.0 GB array per depth.  Compact it at load time; the
            # lookup contract remains identical through CompactStrategy.
            sums = layout.compact_from_dense(sums)
        strategy = CompactStrategy.from_sums(layout, sums)
        return cls(tree, strategy, sampler, iteration=int(payload["iteration"]))

    # -- serving contract ------------------------------------------------------

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        self._raise_fraction = None
        self._raise_target = None
        # Diagnostics describe this decision only.  Retaining a timeout or a
        # successful resolve from an earlier street made preflop/late-street
        # logs falsely report that stale result.
        self.last_continual_search = None
        # Stale diagnostics are worse than none: the decision log would report a
        # resize from an earlier street as if it happened on this one.
        self.last_all_in_rescale = None
        # This strategic restriction must precede continual resolving: the
        # resolver uses a richer exact tree and could otherwise reintroduce a
        # donk bet that the native blueprint deliberately excludes.
        if self.tree.config.no_donk_srp and self._must_check_no_donk(game, player):
            return NEURAL_CHECK_CALL
        searchable = game.street >= 2
        if self.continual_search and game.street in self.continual_streets:
            continual_choice = self._continual_decision(game, player)
            if continual_choice is not None:
                return continual_choice
        elif self.exact_river_search and game.street == 3:
            river_choice = self._exact_river_decision(game, player)
            if river_choice is not None:
                return river_choice
        elif self.subgame_search and searchable:
            subgame_choice = self._subgame_decision(game, player)
            if subgame_choice is not None:
                return subgame_choice
        located = self._locate(game, player)
        if located is None:
            return self._safe_default(game, player)
        node = located
        bucket = self._bucket(game, player, int(self.tree.street[node]))
        if bucket is None:
            return self._safe_default(game, player)
        probabilities = self.strategy[node, bucket]
        actions = [action for action in range(self.tree.config.num_actions) if self.tree.legal[node][action]]
        weights = [max(float(probabilities[action]), 0.0) for action in actions]
        if sum(weights) <= 0:
            weights = [1.0] * len(actions)
        choice = self._rng.choices(actions, weights=weights)[0]
        if choice == FOLD:
            return NEURAL_FOLD
        if choice == CHECK_CALL:
            return NEURAL_CHECK_CALL
        if choice == ALL_IN:
            return self._all_in_choice(game, player, node)
        return self._to_neural_raise(game, player, int(self.tree.street[node]), choice)

    @staticmethod
    def _must_check_no_donk(game: HeadsUpHoldem, player: int) -> bool:
        """Whether ``player`` is the non-aggressor acting first in a SRP.

        The most recent raiser remains the aggressor across streets, so an OOP
        check-raiser is allowed to lead the following street. Opponent donks
        remain observable/off-tree; this guard constrains only our action.
        """
        if game.street <= 0:
            return False
        current_street_actions = [
            event for event in game.public_actions if int(event["street"]) == game.street
        ]
        if current_street_actions:
            return False
        preflop_raises = [
            event
            for event in game.public_actions
            if int(event["street"]) == 0 and event["action"] in {"raise", "all_in"}
        ]
        if len(preflop_raises) != 1:
            return False
        aggressors = [
            event
            for event in game.public_actions
            if int(event["street"]) < game.street and event["action"] in {"raise", "all_in"}
        ]
        if not aggressors or int(aggressors[-1]["player"]) == player:
            return False
        legal = game.legal_actions(player)
        return bool(legal.get("check"))

    def strategy_for_state(self, game: HeadsUpHoldem, player: int) -> dict:
        """Return the mixed strategy for a state without sampling or acting."""
        legal = game.legal_actions(player)
        if not legal:
            raise ValueError("The requested player is not due to act.")

        node = self._locate(game, player)
        bucket = None if node is None else self._bucket(game, player, int(self.tree.street[node]))
        warnings = [
            "Cards and bet sizes are mapped into the champion's trained abstraction."
        ]
        if node is None or bucket is None:
            warnings.append(
                "This history did not map to a trained node; the serving fallback is shown."
            )
            return {
                "exact_match": False,
                "node": None,
                "bucket": None,
                "actions": [self._fallback_query_action(game, player)],
                "warnings": warnings,
            }

        combined: dict[tuple[str, int | None], dict] = {}
        probabilities = self.strategy[node, bucket]
        street = int(self.tree.street[node])
        for choice in range(self.tree.config.num_actions):
            if not self.tree.legal[node][choice]:
                continue
            probability = max(float(probabilities[choice]), 0.0)
            action = self._query_action(game, player, choice, probability, street, node=node)
            key = (action["action"], action["amount"])
            if key in combined:
                combined[key]["probability"] += probability
            else:
                combined[key] = action

        total = sum(action["probability"] for action in combined.values())
        if total <= 0:
            warnings.append("The stored strategy had no probability mass; the serving fallback is shown.")
            actions = [self._fallback_query_action(game, player)]
            exact_match = False
        else:
            actions = list(combined.values())
            for action in actions:
                action["probability"] /= total
            actions.sort(key=lambda action: action["probability"], reverse=True)
            exact_match = True

        return {
            "exact_match": exact_match,
            "node": int(node),
            "bucket": int(bucket),
            "actions": actions,
            "warnings": warnings,
        }

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        # Raises execute at the blueprint's computed chip target directly.
        # Routing through execute_action's fraction mapping re-scales the
        # target into rl_env's legacy PPO-era preflop caps (3-bet <= 2x pot,
        # open <= 3.5bb), which collapsed every 3-bet toward a min-click.
        if choice == NEURAL_RAISE and self._raise_target is not None:
            legal = game.legal_actions(player)
            amount = max(int(legal["raise_min"]), min(int(legal["raise_max"]), self._raise_target))
            game.act(player, "raise", amount)
            return
        if choice == NEURAL_FOLD and game.to_call(player) <= 0:
            # Abstraction/reality mismatch guard: off-tree opponent actions
            # (e.g. a limp mapped to a raise for a no_limp tree) can leave the
            # agent's node believing it faces a bet when the real game does
            # not. Folding when checking is free burns the hand — check.
            game.act(player, "check")
            return
        execute_action(game, player, choice, self._raise_fraction)

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        return None

    def parameter_count(self) -> int:
        return int(self.strategy.size)

    # -- translation ------------------------------------------------------------

    @staticmethod
    def _abstract_seat(game: HeadsUpHoldem, player: int) -> int:
        return 0 if player == game.button else 1

    def _locate(self, game: HeadsUpHoldem, player: int) -> int | None:
        """Walk the flattened tree along the hand's public actions."""
        try:
            abstract_seat = 0 if player == game.button else 1
            node = self.tree.root
            rng = random.Random(game.hand_number * 8191 + len(game.public_actions))
            for event in game.public_actions:
                if event["action"] == "blind":
                    continue
                while self.tree.kind[node] == STREET_END:
                    node = int(self.tree.children[node][0])
                if self.tree.kind[node] != DECISION:
                    return None
                action = self._translate_event(node, game, event, rng)
                child = int(self.tree.children[node][action])
                if child < 0:
                    return None
                node = child
            while self.tree.kind[node] == STREET_END:
                node = int(self.tree.children[node][0])
            if self.tree.kind[node] != DECISION or int(self.tree.actor[node]) != abstract_seat:
                return None
            return node
        except Exception:
            return None

    # -- subgame re-solving ------------------------------------------------------

    @staticmethod
    def _search_uid_for(game: HeadsUpHoldem) -> int:
        """Monotonic live-hand id immune to CPython object-id recycling."""

        uid = getattr(game, "_search_uid", None)
        if uid is None:
            uid = GpuBlueprintAgent._SEARCH_UID = (
                getattr(GpuBlueprintAgent, "_SEARCH_UID", 0) + 1
            )
            try:
                game._search_uid = uid
            except Exception:
                uid = id(game)
        return int(uid)

    def _continual_decision(
        self,
        game: HeadsUpHoldem,
        player: int,
    ) -> int | None:
        """Exact-card turn/river decision from the hand's continual session.

        Returns None to fall back to the frozen blueprint. Failure is sticky for
        the rest of the hand (the session marks itself failed), because resuming
        with a half-advanced belief would condition every later solve on a range
        that never existed.
        """
        started = time.monotonic()
        try:
            from backend.search.continual import (
                FLOP_STREET,
                TURN_STREET,
                register_selected_action,
                resolve_decision,
            )

            # Enter at the FLOP only when the first live decision is actually
            # on the flop. The resolver's rich-to-compact ladder checks the
            # actual mid-street tree before any solver allocation.
            # Re-evaluating flop admission for the first time on the turn is
            # wrong: the larger turn pot can make the old flop look cheap, then
            # opening a new flop session retrospectively pays several catch-up
            # solves for actions that have already happened.
            # Enter no earlier than the shallowest street we actually resolve:
            # entering at the turn while only serving the river would re-derive
            # turn ranges from the blueprint for no benefit and pay a turn solve.
            uid = self._search_uid_for(game)
            key = (uid, int(game.hand_number))
            existing = self._continual_sessions.get(key)
            earliest = min(self.continual_streets)
            if existing is not None:
                entry_street = int(existing.entry_street)
            elif game.street == FLOP_STREET:
                entry_street = max(FLOP_STREET, earliest)
            else:
                # No flop session survived/was opened. Start from the current
                # supported street instead of reconstructing an earlier one.
                entry_street = max(TURN_STREET, earliest)

            solution = resolve_decision(
                self,
                game,
                player,
                key=key,
                sessions=self._continual_sessions,
                iterations=self.continual_iterations,
                budget_ms=self.continual_budget_ms,
                entry_street=entry_street,
            )
            tree = solution.tree
            node = int(solution.node)
            abstract_seat = self._abstract_seat(game, player)
            if tree.kind[node] != DECISION or int(tree.actor[node]) != abstract_seat:
                raise RuntimeError("continual resolve root actor does not match the live player")

            hole = tuple(sorted(card_id(card) for card in game.hole_cards[player]))
            combo_index = _COMBO_INDEX[hole]
            probabilities = solution.strategy[node, combo_index]
            actions = [
                action
                for action in range(tree.config.num_actions)
                if tree.legal[node][action]
            ]
            weights = [max(float(probabilities[action]), 0.0) for action in actions]
            if sum(weights) <= 0:
                raise RuntimeError("continual strategy has no legal probability mass")
            choice = self._rng.choices(actions, weights=weights)[0]
            # Record the exact per-combo likelihood so our own range advances
            # from the policy actually played, not from the blueprint.
            register_selected_action(
                solution,
                event_index=len(game.public_actions),
                actor_seat=abstract_seat,
                action=choice,
            )
            # Record the ACTING mix, not just the chosen action. Without this the
            # decision log can only show the blueprint's distribution, which is
            # not the strategy that played — so "why did it shove?" is
            # unanswerable after the fact.
            fractions = tree.config.fractions(int(game.street))
            labels = ["fold", "check/call", "all-in"] + [f"raise {f}x" for f in fractions]
            acting_mix = [
                {"a": labels[action] if action < len(labels) else str(action),
                 "p": round(float(probabilities[action]), 4)}
                for action in actions
                if float(probabilities[action]) > 1e-4
            ]
            self.last_continual_search = {
                **solution.diagnostics,
                "status": "resolved",
                "node": node,
                "combo": int(combo_index),
                "choice": int(choice),
                "acting_mix": sorted(acting_mix, key=lambda row: -row["p"]),
                "decision_elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            }

            if choice == FOLD:
                return NEURAL_FOLD
            if choice == CHECK_CALL:
                return NEURAL_CHECK_CALL
            if choice == ALL_IN:
                return NEURAL_ALL_IN
            return self._to_neural_raise(game, player, int(game.street), choice, tree=tree)
        except Exception as error:
            resource_diagnostics = getattr(error, "resource_diagnostics", None)
            # A first-decision flop refusal happens before any belief update or
            # CUDA allocation. Drop that untouched session so the hand can
            # still start a clean exact session on the turn instead of losing
            # resolving for every later street.
            if (
                resource_diagnostics is not None
                and game.street == 1
                and "key" in locals()
            ):
                self._continual_sessions.pop(key, None)
            self.last_continual_search = {
                "mode": "continual-exact-v1",
                "status": (
                    "resource-refused"
                    if resource_diagnostics is not None
                    else "blueprint-fallback"
                ),
                "street": int(game.street),
                "error": str(error),
                "resource_admission": resource_diagnostics,
                "decision_elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            }
            return None

    def _exact_river_decision(
        self,
        game: HeadsUpHoldem,
        player: int,
    ) -> int | None:
        """Fresh exact-card safe resolve at the current real river state."""

        started = time.monotonic()
        try:
            from backend.search.exact_river import (
                register_selected_action,
                solve_exact_river,
            )

            uid = self._search_uid_for(game)
            key = (uid, int(game.hand_number))
            solution = solve_exact_river(
                self,
                game,
                player,
                key=key,
                sessions=self._river_sessions,
                iterations=self.exact_river_iterations,
                budget_ms=self.exact_river_budget_ms,
            )
            tree = solution.tree
            node = int(tree.root)
            abstract_seat = self._abstract_seat(game, player)
            if (
                tree.kind[node] != DECISION
                or int(tree.actor[node]) != abstract_seat
            ):
                raise RuntimeError("exact river root actor does not match the live player")

            hole = tuple(sorted(card_id(card) for card in game.hole_cards[player]))
            combo_index = _COMBO_INDEX[hole]
            probabilities = solution.strategy[node, combo_index]
            actions = [
                action
                for action in range(tree.config.num_actions)
                if tree.legal[node][action]
            ]
            weights = [
                max(float(probabilities[action]), 0.0)
                for action in actions
            ]
            if sum(weights) <= 0:
                raise RuntimeError("exact river strategy has no legal probability mass")
            choice = self._rng.choices(actions, weights=weights)[0]
            register_selected_action(
                solution,
                event_index=len(game.public_actions),
                actor_seat=abstract_seat,
                action=choice,
            )
            self.last_river_search = {
                **solution.diagnostics,
                "status": "resolved",
                "node": node,
                "combo": int(combo_index),
                "choice": int(choice),
                "decision_elapsed_ms": round(
                    (time.monotonic() - started) * 1000.0,
                    1,
                ),
            }

            if choice == FOLD:
                return NEURAL_FOLD
            if choice == CHECK_CALL:
                return NEURAL_CHECK_CALL
            if choice == ALL_IN:
                return NEURAL_ALL_IN
            return self._to_neural_raise(game, player, 3, choice, tree=tree)
        except Exception as error:
            # Fail closed: never expose a partial/late solution. The belief
            # session marks itself failed, so the whole remaining hand stays
            # on the frozen blueprint instead of resuming with a false range.
            self.last_river_search = {
                "mode": "exact-card-safe-river-v1",
                "status": "blueprint-fallback",
                "error": str(error),
                "decision_elapsed_ms": round(
                    (time.monotonic() - started) * 1000.0,
                    1,
                ),
            }
            return None

    def _subgame_decision(self, game: HeadsUpHoldem, player: int) -> int | None:
        """Solve (once per street entry) and play from the turn/river subgame."""
        try:
            from backend.search.gpu_subgame import solve_subgame

            abstract_seat = self._abstract_seat(game, player)
            # A per-engine UID keys the cache: eval harnesses build a fresh
            # engine per hand (hand_number always 1), and raw id(game) is NOT
            # safe either — CPython recycles freed addresses, so later hands
            # hit stale solutions from dead engines (found via a 6-minute
            # "search" A/B that never actually searched, 2026-07-24).
            uid = self._search_uid_for(game)
            key = (uid, game.hand_number, len(game.community))
            solution = self._subgame_cache.get(key)
            if solution is None:
                # Reuse a turn solution for river decisions when it exists.
                turn_key = (uid, game.hand_number, 4)
                solution = self._subgame_cache.get(turn_key)
                if solution is None or game.street < 3:
                    if self.safe_search:
                        from backend.search.safe_subgame import solve_subgame_safe as _solve
                    else:
                        _solve = solve_subgame
                    solution = _solve(self, game, player, iterations=self.subgame_iterations)
                    self._subgame_cache[key] = solution
                    if len(self._subgame_cache) > 8:
                        oldest = min(self._subgame_cache)
                        self._subgame_cache.pop(oldest, None)

            tree = solution.tree
            node = tree.root
            rng = random.Random(game.hand_number * 733)
            for event in game.public_actions:
                if event["action"] == "blind" or int(event.get("street", 0)) < tree.start_street:
                    continue
                while tree.kind[node] == STREET_END:
                    node = int(tree.children[node][0])
                if tree.kind[node] != DECISION:
                    return None
                action = self._translate_event(node, game, event, rng, tree=tree)
                child = int(tree.children[node][action])
                if child < 0:
                    return None
                node = child
            while tree.kind[node] == STREET_END:
                node = int(tree.children[node][0])
            if tree.kind[node] != DECISION or int(tree.actor[node]) != abstract_seat:
                return None

            street = int(tree.street[node])
            if street in (1, 2):
                # Flop/turn decision inside a re-solved subgame: buckets come
                # from the solution's own board bucketing (a street-1 node
                # previously fell into the river branch below, silently
                # discarding every flop solve).
                hole = tuple(sorted(card_id(card) for card in game.hole_cards[player]))
                bucket = int(solution.street_buckets[street][_COMBO_INDEX[hole]])
                if bucket < 0:
                    return None
            else:
                bucket = self._bucket(game, player, 3)
                if bucket is None:
                    return None

            probabilities = solution.strategy[node, bucket]
            actions = [action for action in range(tree.config.num_actions) if tree.legal[node][action]]
            weights = [max(float(probabilities[action]), 0.0) for action in actions]
            if sum(weights) <= 0:
                return None
            choice = self._rng.choices(actions, weights=weights)[0]
            if choice == FOLD:
                return NEURAL_FOLD
            if choice == CHECK_CALL:
                return NEURAL_CHECK_CALL
            if choice == ALL_IN:
                return NEURAL_ALL_IN
            return self._to_neural_raise(game, player, street, choice, tree=tree)
        except Exception:
            return None

    def _translate_event(
        self, node: int, game: HeadsUpHoldem, event: dict, rng: random.Random, tree: BettingTree | None = None
    ) -> int:
        tree = tree or self.tree
        legal = tree.legal[node]
        kind = event["action"]
        if kind == "fold":
            return FOLD if legal[FOLD] else CHECK_CALL
        if kind in ("check", "call"):
            if legal[CHECK_CALL]:
                return CHECK_CALL
            # no_limp trees have no open-limp branch; map an opponent's limp
            # to the smallest raise so node tracking survives (off-tree
            # action mapping, same spirit as pseudo-harmonic sizing).
            street = int(tree.street[node])
            fractions = tree.config.fractions(street)
            for index, _ in sorted(enumerate(fractions), key=lambda pair: pair[1]):
                if legal[3 + index]:
                    return 3 + index
            return ALL_IN if legal[ALL_IN] else FOLD
        if kind == "all_in" or event.get("action_index") == 3:
            return ALL_IN if legal[ALL_IN] else CHECK_CALL

        pot_before = float(event.get("pot_before", game.pot))
        to_call_before = float(event.get("to_call_before", 0))
        current_bet_before = float(event.get("current_bet_before", 0))
        pot_after_call = max(pot_before + to_call_before, 1.0)
        observed = max(float(event["amount"]) - current_bet_before, 0.0) / pot_after_call

        street = int(tree.street[node])
        fractions = tree.config.fractions(street)
        raise_ids = [3 + index for index in range(len(fractions)) if legal[3 + index]]
        if not raise_ids:
            return ALL_IN if legal[ALL_IN] and observed > 1.5 else CHECK_CALL
        sized = sorted(raise_ids, key=lambda action: fractions[action - 3])
        below = [action for action in sized if fractions[action - 3] <= observed]
        above = [action for action in sized if fractions[action - 3] >= observed]
        if not below:
            return above[0]
        if not above:
            return below[-1]
        lower, upper = below[-1], above[0]
        if lower == upper:
            return lower
        weight_lower, weight_upper = pseudo_harmonic_weights(observed, fractions[lower - 3], fractions[upper - 3])
        return rng.choices([lower, upper], weights=[weight_lower, weight_upper])[0]

    def _bucket(self, game: HeadsUpHoldem, player: int, street: int) -> int | None:
        hole = tuple(sorted(card_id(card) for card in game.hole_cards[player]))
        if street == 0:
            return preflop_class(hole)
        board = tuple(card_id(card) for card in game.community)[: (0, 3, 4, 5)[street]]
        if len(board) < (0, 3, 4, 5)[street]:
            # Translation drift: the abstract tree is a street ahead of the
            # real board — no valid bucket exists; caller falls back safely.
            return None
        cache_key = (hole, board, street)
        cached = self._equity_cache.get(cache_key)
        if cached is None:
            combo_index = _COMBO_INDEX[hole]
            if street == 3:
                if getattr(self.sampler, "potential_aware", False):
                    cached = self.sampler.street_bucket_for_combo(
                        board,
                        street,
                        combo_index,
                        random.Random(hash((hole, board)) & 0x7FFFFFFF),
                    )
                    if cached is None:
                        return None
                else:
                    # Legacy river: scalar equity quantile.
                    equity = equity_from_scores(score_all_combos(board))[combo_index]
                    if equity < 0:
                        return None
                    counts = self.sampler.bucket_counts()
                    cached = min(int(equity * counts[3]), counts[3] - 1)
            else:
                # Flop/turn: delegate to the sampler's shared bucketing so the
                # served bucket is identical to the trained one (distribution-
                # aware or scalar, whichever the checkpoint used).
                rng = random.Random(hash((hole, board)) & 0x7FFFFFFF)
                cached = self.sampler.street_bucket_for_combo(board, street, combo_index, rng)
                if cached is None:
                    return None
            self._equity_cache[cache_key] = cached
        return int(cached)

    def _abstract_matched_pot(self, node: int) -> float:
        """Matched pot in bb at an abstract node, memoised per node."""
        cached = self._node_pot_cache.get(node)
        if cached is None:
            from backend.search.exact_river import _node_matched_pot

            cached = float(_node_matched_pot(self.tree, node))
            self._node_pot_cache[node] = cached
        return cached

    def _all_in_choice(self, game: HeadsUpHoldem, player: int, node: int) -> int:
        """ALL-IN, resized when translation moved the goalposts.

        ALL-IN is the blueprint's only size-bearing action that used to execute
        literally. Every `raise` action re-derives its chip amount from the REAL
        pot (`_raise_target_for_choice`), but an all-in simply shoved whatever
        was behind. `_locate` matches nodes by translated action sequence and
        never compares pot/stack geometry, so an off-tree opponent line -- a few
        repeated min-raises exhaust the tree's 3-raise cap -- maps a real state
        onto an abstract node whose pot is far larger than the real one. The
        trained action "jam about 2x the pot" then executes as "jam 24x the pot".

        Measured 2026-07-28 (`tools/overbet_audit.py`, 200bb, 200 hands, resolver
        OFF): vs a min-raiser, 8 shoves of 3.3-15.4x pot in 640 decisions, 4 of
        them preflop, every amount within 520 chips of the full stack. Zero
        against a calling station and zero in self-play, because neither line
        exhausts the raise cap. Separately, 77-79% of decision nodes offer no
        raise below all-in and carry ~30% all-in mass, but the abstract jam is
        <=3x pot at 14,492 of those 15,188 nodes -- so the mass is trained and
        reasonable, and only the translation is wrong.

        The repair is the action translation raises already get: preserve the
        size RELATIVE TO THE MATCHED POT that the abstract all-in represented.
        Both ratios are measured the same way -- total chips committed after
        shoving, over the matched pot before it -- so the units cancel and a
        genuinely short stack (where the two agree) still jams unchanged.
        """
        sized = self._all_in_size(game, player, node)
        if sized is None:
            return NEURAL_ALL_IN
        target, diagnostics = sized
        legal = game.legal_actions(player)
        minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
        self._raise_target = target
        self._raise_fraction = (
            0.5 if maximum <= minimum
            else min(0.995, max(0.005, (target - minimum) / (maximum - minimum)))
        )
        self.last_all_in_rescale = diagnostics
        return NEURAL_RAISE

    def _all_in_size(
        self, game: HeadsUpHoldem, player: int, node: int
    ) -> tuple[int, dict] | None:
        """The chip target a translated ALL-IN should really bet, or None to jam.

        Pure: reports the size without touching agent state, so the query and
        decision-log paths show the amount that will actually be wagered.
        """
        if not self.all_in_geometry_guard:
            return None
        legal = game.legal_actions(player)
        if not legal.get("all_in") or not legal.get("raise"):
            return None
        matched_abstract = self._abstract_matched_pot(node)
        matched_real = 2.0 * float(min(game.contributions))
        if matched_abstract <= 0.0 or matched_real <= 0.0:
            return None

        ratio_abstract = float(self.tree.config.stack_bb) / matched_abstract
        ratio_real = (float(game.contributions[player]) + float(game.stacks[player])) / matched_real
        # Commit the same multiple of the matched pot the abstract jam did, and
        # never more than the absolute bound.
        tolerance = float(getattr(self, "all_in_geometry_tolerance", ALL_IN_GEOMETRY_TOLERANCE))
        cap = float(getattr(self, "all_in_max_pot_multiple", ALL_IN_MAX_POT_MULTIPLE))
        # The cap bounds the CORRECTION, never the TRIGGER.  Folding it into the
        # trigger as min(ratio_abstract * tolerance, cap) made the geometry test
        # unreachable whenever the abstract jam was itself larger than the cap --
        # and a preflop jam is inherently ~200x the 1bb matched pot at 200bb.
        # Measured over 150 LBR pairs at 200bb: the guard fired 642 times, and
        # 575 (89.6%) were cap-only firings where geometry was fine, 483 of them
        # with real == abstract EXACTLY.  Trimming a perfectly translated jam is
        # not a translation fix, it is a strategy change, and it is why the guard
        # measured -268.82 bb/100.  Only genuine distortion may fire.
        allowed = ratio_abstract * tolerance
        if ratio_real <= allowed:
            return None  # the jam still means roughly what it was trained to mean

        desired_total = min(ratio_abstract, cap) * matched_real
        already_committed = float(game.contributions[player]) - float(game.round_bets[player])
        minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
        target = max(minimum, min(maximum, int(round(desired_total - already_committed))))
        if target >= maximum:
            return None  # the sized bet IS the stack; nothing to trim
        return target, {
            "node": int(node),
            "abstract_pot_bb": round(matched_abstract, 2),
            "abstract_shove_x_pot": round(ratio_abstract, 2),
            "real_shove_x_pot": round(ratio_real, 2),
            "shove_amount": maximum,
            "sized_to": target,
        }

    def _to_neural_raise(
        self, game: HeadsUpHoldem, player: int, street: int, choice: int, tree: BettingTree | None = None
    ) -> int:
        tree = tree or self.tree
        legal = game.legal_actions(player)
        if not legal.get("raise"):
            return NEURAL_ALL_IN if legal.get("all_in") else NEURAL_CHECK_CALL
        target = self._raise_target_for_choice(game, player, street, choice, tree)
        # Executed directly by execute() — the chip amount the abstract tree
        # actually trained with, not a fraction of someone else's interval.
        self._raise_target = target
        minimum, maximum = float(legal["raise_min"]), float(legal["raise_max"])
        if maximum <= minimum:
            self._raise_fraction = 0.5
        else:
            self._raise_fraction = min(0.995, max(0.005, (target - minimum) / (maximum - minimum)))
        return NEURAL_RAISE

    @staticmethod
    def _raise_target_for_choice(
        game: HeadsUpHoldem,
        player: int,
        street: int,
        choice: int,
        tree: BettingTree,
    ) -> int:
        legal = game.legal_actions(player)
        to_call = float(legal["to_call"])
        fraction = tree.config.fractions(street)[choice - 3]
        raise_by = fraction * (game.pot + to_call)
        target = float(legal["player_bet"]) + to_call + raise_by
        return max(int(legal["raise_min"]), min(int(legal["raise_max"]), int(round(target))))

    def _query_action(
        self,
        game: HeadsUpHoldem,
        player: int,
        choice: int,
        probability: float,
        street: int,
        node: int | None = None,
    ) -> dict:
        legal = game.legal_actions(player)
        if choice == FOLD:
            if legal.get("fold"):
                action, amount = "fold", None
            else:
                action = "check" if legal.get("check") else "call"
                amount = int(legal.get("to_call", 0)) if action == "call" else None
        elif choice == CHECK_CALL:
            action = "check" if legal.get("check") else "call"
            amount = int(legal.get("to_call", 0)) if action == "call" else None
        elif choice == ALL_IN:
            if legal.get("all_in"):
                # Report the resized amount when translation distorted the jam,
                # so the query shows what would actually be wagered.
                sized = None if node is None else self._all_in_size(game, player, node)
                if sized is not None:
                    action, amount = "raise", int(sized[0])
                else:
                    action = "all_in"
                    amount = int(legal.get("raise_max", 0))
            else:
                action = "check" if legal.get("check") else "call"
                amount = int(legal.get("to_call", 0)) if action == "call" else None
        else:
            if legal.get("raise"):
                action = "raise"
                amount = self._raise_target_for_choice(game, player, street, choice, self.tree)
            elif legal.get("all_in"):
                action = "all_in"
                amount = int(legal.get("raise_max", 0))
            else:
                action = "check" if legal.get("check") else "call"
                amount = int(legal.get("to_call", 0)) if action == "call" else None
        label = action.replace("_", " ").title()
        if action == "raise" and amount is not None:
            label = f"Raise to {amount:,}"
        elif action == "call" and amount is not None:
            label = f"Call {amount:,}"
        return {
            "action": action,
            "label": label,
            "amount": amount,
            "probability": float(probability),
        }

    def _fallback_query_action(self, game: HeadsUpHoldem, player: int) -> dict:
        choice = self._safe_default(game, player)
        legal = game.legal_actions(player)
        if choice == NEURAL_FOLD:
            action, amount = "fold", None
        elif choice == NEURAL_CHECK_CALL:
            action = "check" if legal.get("check") else "call"
            amount = int(legal.get("to_call", 0)) if action == "call" else None
        else:
            action = "all_in"
            amount = int(legal.get("raise_max", 0))
        label = action.replace("_", " ").title()
        if action == "call" and amount is not None:
            label = f"Call {amount:,}"
        return {"action": action, "label": label, "amount": amount, "probability": 1.0}

    @staticmethod
    def _safe_default(game: HeadsUpHoldem, player: int) -> int:
        legal = game.legal_actions(player)
        if legal.get("check") or legal.get("call"):
            return NEURAL_CHECK_CALL
        if legal.get("all_in"):
            return NEURAL_ALL_IN
        return NEURAL_FOLD
