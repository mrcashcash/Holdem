"""Exact CFR+ for a documented, solvable heads-up poker abstraction.

This module intentionally does *not* claim to solve no-limit Hold'em.  It
solves a small extensive-form game with five private-strength buckets, public
street/SPR/texture context, and the production semantic action set.  Unlike the
legacy hand-authored oracle, every chance outcome, legal action, continuation,
and terminal chip payoff is represented in the game tree.  Its NashConv metric
therefore applies to this abstraction only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .poker import HeadsUpHoldem
from .rl_env import ACTION_COUNT


STRENGTH_BUCKETS = 5
CONTEXTS = tuple((street, spr, texture) for street in range(4) for spr in range(4) for texture in range(3))
_STACK_BY_SPR = (1.0, 1.5, 2.5, 4.0)


@dataclass(frozen=True)
class AbstractionAudit:
    """Exact-game diagnostics; they do not measure Hold'em exploitability."""

    iterations: int
    average_value: float
    nash_conv: float
    average_positive_regret: float
    information_sets: int


@dataclass
class SolverTeacherRecord:
    """Low-confidence strategy target projected onto a real legal action mask."""

    observation: list[float]
    mask: list[bool]
    strategy: list[float]
    value: float
    confidence: float
    street: int

    def payload(self) -> dict:
        return {
            "observation": self.observation,
            "mask": self.mask,
            "strategy": self.strategy,
            "value": self.value,
            "confidence": self.confidence,
            "street": self.street,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> SolverTeacherRecord:
        return cls(
            list(payload["observation"]),
            list(payload["mask"]),
            list(payload["strategy"]),
            float(payload.get("value", 0.0)),
            float(payload.get("confidence", 0.0)),
            int(payload.get("street", 0)),
        )


class HoldemAbstractionCfr:
    """Full-tree CFR+ over a compact, explicitly defined poker abstraction."""

    _TERMINALS = frozenset({"11", "20", "21", "30", "31", "120", "121", "130", "131", "230", "231", "1230", "1231"})
    _NODE_PLAYERS = {"": 0, "1": 1, "2": 1, "3": 1, "12": 0, "13": 0, "23": 0, "123": 1}
    _LEGAL_ACTIONS = {
        "": (1, 2, 3),          # check, bet, all-in
        "1": (1, 2, 3),         # check-back, bet, all-in
        "2": (0, 1, 3),         # fold, call, re-shove
        "3": (0, 1),            # fold, call all-in
        "12": (0, 1, 3),        # fold, call, re-shove
        "13": (0, 1),           # fold, call all-in
        "23": (0, 1),           # fold, call all-in
        "123": (0, 1),          # fold, call re-shove
    }

    def __init__(self) -> None:
        self.regrets: dict[str, list[float]] = {}
        self.strategy_sums: dict[str, list[float]] = {}
        self.iterations = 0

    @classmethod
    def _node_player(cls, history: str) -> int:
        return cls._NODE_PLAYERS[history]

    @classmethod
    def _legal_actions(cls, history: str) -> tuple[int, ...]:
        return cls._LEGAL_ACTIONS[history]

    @staticmethod
    def _key(context: tuple[int, int, int], player: int, strength: int, history: str) -> str:
        street, spr, texture = context
        return f"{street}:{spr}:{texture}:{player}:{strength}:{history or '-'}"

    @staticmethod
    def _regret_strategy(regrets: list[float], legal: tuple[int, ...]) -> list[float]:
        positive = [max(0.0, regrets[action]) for action in legal]
        total = sum(positive)
        if total <= 1e-12:
            return [1.0 / len(legal) if action in legal else 0.0 for action in range(ACTION_COUNT)]
        return [max(0.0, regrets[action]) / total if action in legal else 0.0 for action in range(ACTION_COUNT)]

    def _current_strategy(self, context: tuple[int, int, int], player: int, strength: int, history: str) -> list[float]:
        legal = self._legal_actions(history)
        key = self._key(context, player, strength, history)
        return self._regret_strategy(self.regrets.setdefault(key, [0.0] * ACTION_COUNT), legal)

    def _average_strategy(self, context: tuple[int, int, int], player: int, strength: int, history: str) -> list[float]:
        legal = self._legal_actions(history)
        sums = self.strategy_sums.get(self._key(context, player, strength, history), [0.0] * ACTION_COUNT)
        total = sum(sums[action] for action in legal)
        if total <= 1e-12:
            return [1.0 / len(legal) if action in legal else 0.0 for action in range(ACTION_COUNT)]
        return [sums[action] / total if action in legal else 0.0 for action in range(ACTION_COUNT)]

    def _terminal_utility(self, context: tuple[int, int, int], strengths: tuple[int, int], history: str) -> float:
        """Return player-zero net chips with blinds and all bet commitments explicit."""
        _, spr, _ = context
        all_in_total = 0.5 + _STACK_BY_SPR[spr]
        commits = [0.5, 0.5]
        prefix = ""
        for action_text in history:
            player = self._node_player(prefix)
            action = int(action_text)
            if action == 2:
                commits[player] = min(all_in_total, max(commits) + 1.0)
            elif action == 3:
                commits[player] = all_in_total
            elif action == 1 and commits[player] < max(commits):
                commits[player] = max(commits)
            prefix += action_text
        if history[-1] == "0":
            winner = 1 - self._node_player(history[:-1])
        elif strengths[0] > strengths[1]:
            winner = 0
        elif strengths[0] < strengths[1]:
            winner = 1
        else:
            return 0.5 * (commits[1] - commits[0])
        return commits[1] if winner == 0 else -commits[0]

    def _cfr(self, context: tuple[int, int, int], strengths: tuple[int, int], history: str, reach_zero: float, reach_one: float) -> float:
        if history in self._TERMINALS:
            return self._terminal_utility(context, strengths, history)
        player = self._node_player(history)
        strength = strengths[player]
        legal = self._legal_actions(history)
        key = self._key(context, player, strength, history)
        strategy = self._current_strategy(context, player, strength, history)
        strategy_sum = self.strategy_sums.setdefault(key, [0.0] * ACTION_COUNT)
        own_reach = reach_zero if player == 0 else reach_one
        for action in legal:
            strategy_sum[action] += own_reach * strategy[action]
        utilities: dict[int, float] = {}
        for action in legal:
            if player == 0:
                utilities[action] = self._cfr(context, strengths, history + str(action), reach_zero * strategy[action], reach_one)
            else:
                utilities[action] = self._cfr(context, strengths, history + str(action), reach_zero, reach_one * strategy[action])
        node_value = sum(strategy[action] * utilities[action] for action in legal)
        opponent_reach = reach_one if player == 0 else reach_zero
        regrets = self.regrets.setdefault(key, [0.0] * ACTION_COUNT)
        for action in legal:
            regret = utilities[action] - node_value if player == 0 else node_value - utilities[action]
            regrets[action] = max(0.0, regrets[action] + opponent_reach * regret)
        return node_value

    def solve(self, iterations: int = 1) -> None:
        """Run exact chance enumeration and full-tree CFR+ iterations."""
        for _ in range(max(1, iterations)):
            for context in CONTEXTS:
                for strength_zero in range(STRENGTH_BUCKETS):
                    for strength_one in range(STRENGTH_BUCKETS):
                        self._cfr(context, (strength_zero, strength_one), "", 1.0, 1.0)
            self.iterations += 1

    def _evaluate_average(self, context: tuple[int, int, int], strengths: tuple[int, int], history: str) -> float:
        if history in self._TERMINALS:
            return self._terminal_utility(context, strengths, history)
        player = self._node_player(history)
        strategy = self._average_strategy(context, player, strengths[player], history)
        return sum(strategy[action] * self._evaluate_average(context, strengths, history + str(action)) for action in self._legal_actions(history))

    def _best_response(self, context: tuple[int, int, int], hero: int, own_strength: int, history: str, posterior: tuple[float, ...]) -> float:
        """Exact behavioural best response with Bayesian updates after observed actions."""
        if history in self._TERMINALS:
            return sum(
                probability * self._terminal_utility(context, (own_strength, other) if hero == 0 else (other, own_strength), history)
                for other, probability in enumerate(posterior)
            )
        player = self._node_player(history)
        legal = self._legal_actions(history)
        if player == hero:
            responses = [self._best_response(context, hero, own_strength, history + str(action), posterior) for action in legal]
            # Utilities are always expressed for player zero. Player one's
            # behavioural best response must therefore minimize that value.
            return max(responses) if hero == 0 else min(responses)
        action_mass: dict[int, float] = {}
        for action in legal:
            action_mass[action] = sum(
                probability * self._average_strategy(context, player, other, history)[action]
                for other, probability in enumerate(posterior)
            )
        value = 0.0
        for action, mass in action_mass.items():
            if mass <= 1e-12:
                continue
            updated = tuple(
                probability * self._average_strategy(context, player, other, history)[action] / mass
                for other, probability in enumerate(posterior)
            )
            value += mass * self._best_response(context, hero, own_strength, history + str(action), updated)
        return value

    def audit(self) -> AbstractionAudit:
        total_value = 0.0
        br_zero = 0.0
        br_one = 0.0
        uniform = tuple(1.0 / STRENGTH_BUCKETS for _ in range(STRENGTH_BUCKETS))
        for context in CONTEXTS:
            for strength_zero in range(STRENGTH_BUCKETS):
                for strength_one in range(STRENGTH_BUCKETS):
                    total_value += self._evaluate_average(context, (strength_zero, strength_one), "")
            br_zero += sum(self._best_response(context, 0, strength, "", uniform) for strength in range(STRENGTH_BUCKETS)) / STRENGTH_BUCKETS
            br_one += sum(self._best_response(context, 1, strength, "", uniform) for strength in range(STRENGTH_BUCKETS)) / STRENGTH_BUCKETS
        average_value = total_value / (len(CONTEXTS) * STRENGTH_BUCKETS * STRENGTH_BUCKETS)
        scale = len(CONTEXTS)
        positive = [max(0.0, value) for regrets in self.regrets.values() for value in regrets]
        return AbstractionAudit(
            self.iterations,
            average_value,
            max(0.0, (br_zero - br_one) / max(1, scale)),
            sum(positive) / max(1, len(positive) * max(1, self.iterations)),
            len(self.regrets),
        )

    @staticmethod
    def _context_from_game(game: HeadsUpHoldem) -> tuple[int, int, int]:
        spr = min(3, int(min(game.stacks) / max(1, game.pot + game.big_blind)))
        board_suits = [card[1] for card in game.community]
        texture = 2 if board_suits and max(board_suits.count(suit) for suit in set(board_suits)) >= 3 else 1 if len({card[0] for card in game.community}) < len(game.community) else 0
        return game.street, spr, texture

    @staticmethod
    def _strength_from_game(game: HeadsUpHoldem, player: int) -> int:
        ranks = sorted((card[0] for card in game.hole_cards[player]), reverse=True)
        raw = (ranks[0] + ranks[1]) / 28
        raw += 0.26 if ranks[0] == ranks[1] else 0.0
        raw += 0.06 if game.hole_cards[player][0][1] == game.hole_cards[player][1][1] else 0.0
        raw += 0.10 * sum(rank in {card[0] for card in game.community} for rank in ranks)
        return min(STRENGTH_BUCKETS - 1, max(0, int(raw * STRENGTH_BUCKETS)))

    def _history_from_game(self, game: HeadsUpHoldem, player: int) -> str:
        actions = [int(event.get("action_index", -1)) for event in game.public_actions if int(event.get("street", -1)) == game.street and int(event.get("action_index", -1)) >= 0]
        history = ""
        for action in actions:
            candidate = history + str(action)
            if candidate not in self._NODE_PLAYERS and candidate not in self._TERMINALS:
                break
            history = candidate
            if history in self._TERMINALS:
                break
        if history in self._NODE_PLAYERS and self._node_player(history) == player:
            return history
        return "" if player == 0 else "1"

    def target(self, game: HeadsUpHoldem, player: int, mask: list[bool], observation: list[float]) -> SolverTeacherRecord:
        """Project the solved abstraction onto a real decision without changing live play."""
        context = self._context_from_game(game)
        strength = self._strength_from_game(game, player)
        history = self._history_from_game(game, player)
        strategy = self._average_strategy(context, player, strength, history)
        projected = [weight if allowed else 0.0 for weight, allowed in zip(strategy, mask)]
        total = sum(projected)
        if total <= 1e-12:
            legal = [index for index, allowed in enumerate(mask) if allowed]
            projected = [1.0 / len(legal) if index in legal else 0.0 for index in range(ACTION_COUNT)] if legal else [0.0] * ACTION_COUNT
        else:
            projected = [weight / total for weight in projected]
        expected = sum(
            self._evaluate_average(context, (strength, other) if player == 0 else (other, strength), history)
            for other in range(STRENGTH_BUCKETS)
        ) / STRENGTH_BUCKETS
        if player == 1:
            expected = -expected
        confidence = min(0.12, 0.02 + 0.10 * self.iterations / (self.iterations + 48))
        return SolverTeacherRecord(observation, mask, projected, expected, confidence, game.street)

    def snapshot(self) -> dict:
        return {"iterations": self.iterations, "regrets": self.regrets, "strategy_sums": self.strategy_sums}

    def restore(self, payload: dict) -> None:
        self.iterations = max(0, int(payload.get("iterations", 0)))
        self.regrets = {
            str(key): [float(value) for value in values]
            for key, values in payload.get("regrets", {}).items()
            if isinstance(values, list) and len(values) == ACTION_COUNT
        }
        self.strategy_sums = {
            str(key): [float(value) for value in values]
            for key, values in payload.get("strategy_sums", {}).items()
            if isinstance(values, list) and len(values) == ACTION_COUNT
        }
