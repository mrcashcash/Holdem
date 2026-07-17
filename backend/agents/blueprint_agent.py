"""Serve the MCCFR blueprint through the live game's agent contract.

The agent mirrors the real ``HeadsUpHoldem`` hand into the abstract game by
replaying the public betting history: every real action is translated onto
the abstract action menu (raises via pseudo-harmonic weights), the real
board cards are injected at chance nodes, and the blueprint's average
strategy is sampled at the resulting infoset. The abstract seat convention
is button = player 0, so engine seats are remapped every hand.

Falls back to check/call if the abstract and real state machines ever
disagree (they can, because the abstraction caps raise counts) — the engine
remains the single source of truth for legality.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from backend.abstraction.actions import ALL_IN, CHECK_CALL, FOLD
from backend.abstraction.buckets import CardAbstraction
from backend.abstraction.actions import pseudo_harmonic_weights
from backend.poker import HeadsUpHoldem
from backend.rl_env import execute_action
from backend.solver import blueprint as blueprint_module
from backend.solver.holdem import _STREET_BOARD, AbstractHoldem, HoldemState
from backend.solver.mccfr import StrategyTable
from backend.vectorized_engine import card_id

NEURAL_FOLD, NEURAL_CHECK_CALL, NEURAL_RAISE, NEURAL_ALL_IN = 0, 1, 2, 3


class BlueprintAgent:
    """Drop-in replacement for ``NeuralAgent`` on the serving path."""

    def __init__(
        self,
        game: AbstractHoldem,
        table: StrategyTable,
        river_search: bool = True,
        river_iterations: int = 400,
    ) -> None:
        self.game = game
        self.table = table
        self.ready = True
        self.river_search = river_search
        self.river_iterations = river_iterations
        self._raise_fraction: float | None = None
        self._rng = random.Random(2026)

    @classmethod
    def try_load(
        cls,
        blueprint_path: Path | None = None,
        abstraction_path: Path | None = None,
    ) -> "BlueprintAgent | None":
        """Load the blueprint artifacts if both exist; otherwise return None."""
        import pickle

        blueprint_file = blueprint_path or blueprint_module.BLUEPRINT_PATH
        abstraction_file = abstraction_path or blueprint_module.ABSTRACTION_PATH
        if not blueprint_file.exists() or not abstraction_file.exists():
            return None
        abstraction = CardAbstraction.load(abstraction_file)
        # Local trainer artifact only (see StrategyTable.save trust boundary).
        with open(blueprint_file, "rb") as handle:
            payload = pickle.load(handle)
        game = AbstractHoldem(abstraction, stack_bb=blueprint_module.STACK_BB)
        return cls(game, payload["table"])

    # -- serving contract ----------------------------------------------------

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        self._raise_fraction = None
        state = self._mirror_state(game, player)
        abstract_seat = self._abstract_seat(game, player)
        if state is None or state.is_terminal() or state.is_chance() or state.current_player() != abstract_seat:
            return self._safe_default(game, player)

        if self.river_search and game.street == 3:
            river_choice = self._river_decision(game, player)
            if river_choice is not None:
                return river_choice

        actions = list(state.legal_actions())
        probabilities = self.table.average_strategy(state.infoset_key(), actions)
        choice = self._rng.choices(actions, weights=list(probabilities))[0]

        if choice == FOLD:
            return NEURAL_FOLD
        if choice == CHECK_CALL:
            return NEURAL_CHECK_CALL
        if choice == ALL_IN:
            return NEURAL_ALL_IN
        return self._to_neural_raise(game, player, state, choice)

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        execute_action(game, player, choice, self._raise_fraction)

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        return None

    def parameter_count(self) -> int:
        return len(self.table)

    # -- translation ----------------------------------------------------------

    @staticmethod
    def _abstract_seat(game: HeadsUpHoldem, player: int) -> int:
        return 0 if player == game.button else 1

    def _mirror_state(self, game: HeadsUpHoldem, player: int) -> HoldemState | None:
        """Replay the hand's public actions inside the abstract game."""
        try:
            abstract_seat = self._abstract_seat(game, player)
            hole = tuple(card_id(card) for card in game.hole_cards[player])
            board_ids = [card_id(card) for card in game.community]
            dummy = [card for card in range(52) if card not in set(hole) | set(board_ids)][:2]
            holes = (hole, tuple(dummy)) if abstract_seat == 0 else (tuple(dummy), hole)

            state = self.game.initial_state()
            state = replace(state, hole=holes)
            translation_rng = random.Random(game.hand_number * 7919 + len(game.public_actions))

            for event in game.public_actions:
                if event["action"] == "blind":
                    continue
                state = self._inject_board(state, board_ids)
                if state.is_terminal() or state.is_chance():
                    return state
                abstract_action = self._translate_event(state, game, event, translation_rng)
                state = state.child(abstract_action)
            return self._inject_board(state, board_ids)
        except Exception:
            return None

    @staticmethod
    def _inject_board(state: HoldemState, board_ids: list[int]) -> HoldemState:
        """Reveal real board cards wherever the abstract game expects chance."""
        while state.is_chance() and state.hole is not None:
            if state.in_runout:
                known = board_ids[len(state.board) :]
                if len(state.board) + len(known) < 5:
                    break  # runout cards not dealt yet in the real game
                return replace(
                    state, board=state.board + tuple(known), street=3, in_runout=False, showdown=True
                )
            needed = _STREET_BOARD[state.street] - len(state.board)
            if needed <= 0 or len(board_ids) < len(state.board) + needed:
                break
            revealed = tuple(board_ids[len(state.board) : len(state.board) + needed])
            state = replace(state, board=state.board + revealed)
        return state

    def _translate_event(
        self,
        state: HoldemState,
        game: HeadsUpHoldem,
        event: dict,
        rng: random.Random,
    ) -> int:
        menu = list(state.legal_actions())
        kind = event["action"]
        if kind == "fold":
            return FOLD if FOLD in menu else CHECK_CALL
        if kind in ("check", "call"):
            return CHECK_CALL
        if kind == "all_in" or event.get("action_index") == 3:
            return ALL_IN if ALL_IN in menu else CHECK_CALL

        # A normal raise: express the raise-to amount as a pot fraction and map
        # it onto the neighbouring menu sizes with pseudo-harmonic weights.
        pot_before = float(event.get("pot_before", game.pot))
        to_call_before = float(event.get("to_call_before", 0))
        current_bet_before = float(event.get("current_bet_before", 0))
        pot_after_call = max(pot_before + to_call_before, 1.0)
        observed_fraction = max(float(event["amount"]) - current_bet_before, 0.0) / pot_after_call

        raise_ids = [action for action in menu if action >= 3]
        if not raise_ids:
            return ALL_IN if ALL_IN in menu and observed_fraction > 1.5 else CHECK_CALL
        fractions = self.game.actions.fractions_for_street(state.street)
        sized = sorted(raise_ids, key=lambda action: fractions[action - 3])
        below = [action for action in sized if fractions[action - 3] <= observed_fraction]
        above = [action for action in sized if fractions[action - 3] >= observed_fraction]
        if not below:
            return above[0]
        if not above:
            return below[-1]
        lower, upper = below[-1], above[0]
        if lower == upper:
            return lower
        weight_lower, weight_upper = pseudo_harmonic_weights(
            observed_fraction, fractions[lower - 3], fractions[upper - 3]
        )
        return rng.choices([lower, upper], weights=[weight_lower, weight_upper])[0]

    def _to_neural_raise(self, game: HeadsUpHoldem, player: int, state: HoldemState, choice: int) -> int:
        legal = game.legal_actions(player)
        if not legal.get("raise"):
            return NEURAL_ALL_IN if legal.get("all_in") else NEURAL_CHECK_CALL
        to_call = float(legal["to_call"])
        fraction = self.game.actions.fractions_for_street(state.street)[choice - 3]
        raise_by = fraction * (game.pot + to_call)
        target = float(legal["player_bet"]) + to_call + raise_by
        minimum, maximum = float(legal["raise_min"]), float(legal["raise_max"])
        if maximum <= minimum:
            self._raise_fraction = 0.5
        else:
            self._raise_fraction = min(0.995, max(0.005, (target - minimum) / (maximum - minimum)))
        return NEURAL_RAISE

    # -- river re-solving ------------------------------------------------------

    def _river_decision(self, game: HeadsUpHoldem, player: int) -> int | None:
        """Re-solve the actual river subgame and act from the solution.

        Unsafe re-solving: both ranges are inferred from the blueprint along
        the public history (see backend/search). Falls back to blueprint play
        (returning None) on any inconsistency.
        """
        try:
            from backend.search.ranges import blueprint_range
            from backend.search.river import RiverSubgame, solve_river

            abstract_seat = self._abstract_seat(game, player)
            opponent = 1 - player
            our_hole = tuple(sorted(card_id(card) for card in game.hole_cards[player]))

            our_range = blueprint_range(self, game, player)
            their_range = blueprint_range(self, game, opponent, extra_blocked=our_hole)
            if our_hole not in our_range:
                our_range[our_hole] = max(our_range.values(), default=1.0) * 0.05

            bb = float(game.big_blind)
            pot_start = sum(game.contributions[side] - game.round_bets[side] for side in (0, 1)) / bb
            stacks_by_seat = [0.0, 0.0]
            ranges_by_seat: list[dict] = [{}, {}]
            for engine_player in (0, 1):
                seat = self._abstract_seat(game, engine_player)
                stacks_by_seat[seat] = (game.stacks[engine_player] + game.round_bets[engine_player]) / bb
                ranges_by_seat[seat] = our_range if engine_player == player else their_range

            subgame = RiverSubgame(
                board=tuple(card_id(card) for card in game.community),
                pot_start=pot_start,
                stacks=(stacks_by_seat[0], stacks_by_seat[1]),
                ranges=(ranges_by_seat[0], ranges_by_seat[1]),
                actions=self.game.actions,
            )
            solver = solve_river(subgame, iterations=self.river_iterations, seed=game.hand_number)

            # Walk the solved tree along the river actions already taken.
            from dataclasses import replace as dc_replace

            dummy = tuple(card for card in range(52) if card not in set(our_hole) | {card_id(c) for c in game.community})[:2]
            combos = (our_hole, dummy) if abstract_seat == 0 else (dummy, our_hole)
            state = dc_replace(subgame.initial_state(), combos=combos)
            translation_rng = random.Random(game.hand_number * 31337)
            for event in game.public_actions:
                if event["action"] == "blind" or int(event.get("street", 0)) != 3:
                    continue
                if state.is_terminal():
                    return None
                abstract_action = self._translate_event(state, game, event, translation_rng)
                state = state.child(abstract_action)
            if state.is_terminal() or state.current_player() != abstract_seat:
                return None

            actions = list(state.legal_actions())
            probabilities = solver.table.average_strategy(state.infoset_key(), actions)
            choice = self._rng.choices(actions, weights=list(probabilities))[0]
            if choice == FOLD:
                return NEURAL_FOLD
            if choice == CHECK_CALL:
                return NEURAL_CHECK_CALL
            if choice == ALL_IN:
                return NEURAL_ALL_IN
            return self._to_neural_raise(game, player, state, choice)
        except Exception:
            return None

    @staticmethod
    def _safe_default(game: HeadsUpHoldem, player: int) -> int:
        legal = game.legal_actions(player)
        if legal.get("check") or legal.get("call"):
            return NEURAL_CHECK_CALL
        if legal.get("all_in"):
            return NEURAL_ALL_IN
        return NEURAL_FOLD
