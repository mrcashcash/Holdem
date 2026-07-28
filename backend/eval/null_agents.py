"""Deliberately trivial agents whose value is analytically known.

These exist to test *instruments*, not to play poker. Every harness in
`backend/eval/` must be able to reproduce a known number before its output
about a real agent is believed (`docs/STATUS.md` §4: three separate harness
bugs shipped numbers that looked plausible for weeks).

The useful property of `always-fold` is that its result is exact rather than
statistical: a player who folds at its first opportunity loses precisely its
posted blind, every hand, with zero variance. As the button/small blind that is
exactly -0.5 bb per hand. Any harness reading something else has a
baseline/accounting bug — which is exactly the shape of the +75 bb/100
inflation that survived ten promotion gates.

`always-call` (a calling station) and `always-min-raise` (a maniac) are
maximally exploitable in opposite directions, which makes them the reference
targets for validating an exploitability probe such as LBR: a working probe
must find a very large exploit against both, and a much smaller one against a
trained blueprint.
"""

from __future__ import annotations

from backend.poker import HeadsUpHoldem

# The serving agent contract's action ids (backend/agents/gpu_blueprint_agent).
FOLD, CHECK_CALL, RAISE, ALL_IN = 0, 1, 2, 3

POLICIES = ("always-fold", "always-call", "always-min-raise", "always-all-in")


class ScriptedAgent:
    """Fixed policy exposing the serving agent contract (select + execute).

    Implements only what evaluation harnesses call: `select`, `execute`, and
    the no-op `observe_completed_hand`. It has no strategy table, so
    `strategy_for_state` is intentionally absent — harnesses that need it must
    tolerate its absence (the duel diagnostics already guard for that).
    """

    def __init__(self, policy: str = "always-fold") -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        self.policy = policy
        self.ready = True
        # Serving agents expose these; harnesses read them for diagnostics.
        self.iteration = 0
        self.subgame_search = False

    # -- serving contract ------------------------------------------------------

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        legal = game.legal_actions(player)
        if not legal:
            raise ValueError("The requested player is not due to act.")
        if self.policy == "always-fold":
            # Folding is illegal when checking is free; a free check does not
            # cost the blind, so the analytic value is unchanged.
            return FOLD if legal.get("fold") else CHECK_CALL
        if self.policy == "always-call":
            return CHECK_CALL
        if self.policy == "always-all-in":
            return ALL_IN if legal.get("all_in") else CHECK_CALL
        return RAISE if legal.get("raise") else CHECK_CALL

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        legal = game.legal_actions(player)
        if choice == FOLD and legal.get("fold"):
            game.act(player, "fold")
        elif choice == ALL_IN and legal.get("all_in"):
            game.act(player, "all_in")
        elif choice == RAISE and legal.get("raise"):
            game.act(player, "raise", int(legal["raise_min"]))
        elif legal.get("check"):
            game.act(player, "check")
        elif legal.get("call"):
            game.act(player, "call")
        else:  # pragma: no cover - the engine always leaves one of the above
            game.act(player, "fold")

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        return None

    def parameter_count(self) -> int:
        return 0


def expected_bb_per_hand(policy: str, as_button: bool, big_blind: float = 100.0, small_blind: float = 50.0) -> float | None:
    """Analytic per-hand result, where one exists.

    Only `always-fold` from the button is exactly determined: the player posts
    the small blind and folds, losing it, regardless of what the opponent does.
    Out of position the result depends on how often the opponent opens, so no
    exact value exists — the caller should bound it instead (see
    `FOLD_OUT_OF_POSITION_BOUNDS`).
    """
    if policy == "always-fold" and as_button:
        return -small_blind / big_blind
    return None


# Out of position there is no exact value. always-fold cannot fold when
# checking is free, so an opponent limp lets the hand continue and even reach a
# showdown it may win. It never voluntarily puts in a chip beyond the posted big
# blind, though, so per hand it can lose at most its own big blind and win at
# most the opponent's matched big blind — bounds that hold for ANY opponent.
FOLD_OUT_OF_POSITION_BOUNDS = (-1.0, 1.0)
