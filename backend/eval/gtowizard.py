"""Benchmark the agent against GTO Wizard AI (the Ruse AI Researcher API).

Why this instrument exists, when Slumbot already does: Slumbot is a 2018
abstraction agent that does not probe, so `docs/STATUS.md` correctly demotes it to
a sanity check and makes LBR the north star. GTO Wizard AI is a real-time
resolving agent that beat Slumbot by 19.4 +- 4.1 bb/100 over 150,000 hands, and
the API returns **AIVAT-adjusted** results, which the paper reports reaches the
same significance with ten times fewer hands. So this is the first external
instrument here that is both stronger than the agent and variance-reduced.

Protocol (https://researcher.gtowizard.com/openapi.json), verified live:

    POST /hands            {"game_name": "HUNL 200BB"}  -> GameServiceResponse
    POST /hands/{id}/act   {"action": "f|k|c|b", "amount": int|null}
    GET  /results?game_name=...                         -> PlayerStatistics

Facts worth pinning, because guessing any of them silently corrupts the run:

* `blinds` is **[big, small]** -- [100.0, 50.0] -- not the usual [small, big].
* `starting_stack` 20,000 = 200bb, and `stack_reset_per_hand` is true, so every
  hand is independent and depth is always exactly 200bb.
* Positions are "SB" and "BB". Heads-up, the SB **is** the button and acts first
  preflop, so hero-is-button iff hero's position is "SB".
* A `b` amount is the **street-cumulative raise-to total**, which is exactly the
  engine's `amount` semantics for a raise -- so `encode_action` needs no rescaling.
* `action_history` is a LIST, and `"_"` marks end-of-round. The engine advances
  streets by itself, so `"_"` is skipped rather than applied.
* `winnings` is raw chips from the hero's perspective; `aivat_score` is the
  luck-adjusted figure and is the one that counts. A verified example: folding the
  BB to a 225 open read winnings -100.0 and aivat_score -14.66.

Design rules inherited from `slumbot.py`, each load-bearing:

1. **The AGENT's own action mapping is used**, never a reimplementation: the hand
   is mirrored into a real `HeadsUpHoldem` and the agent's own `select()` /
   `execute()` choose and apply the move. A separate mapping is a second
   implementation that can silently disagree with the served one.
2. **Position comes from the protocol**, not a constant.
3. **Broken hands are excluded and counted, never folded and scored** -- folding a
   mirroring failure would report a real loss caused by our own bug.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import urllib.error
import urllib.request

from backend.eval.slumbot import _RANK_MAP, _SUIT_MAP
from backend.poker import HeadsUpHoldem, new_deck

BASE_URL = "https://researcher.gtowizard.com"
GAME_NAME = "HUNL 200BB"
#: Cloudflare blocks the default urllib signature; see _request.
USER_AGENT = "holdem-research-agent/1.0 (+https://github.com/mrcashcash)"
#: Confirmed from GET /hands: starting_stack 20,000 with blinds [big, small].
STACK = 20_000
BIG_BLIND = 100
SMALL_BLIND = 50

#: Published anchors (arXiv 2603.23660 Table 2) for validating this harness
#: before any number from it is believed. AIVAT-adjusted bb/100, +- 95% CI.
PUBLISHED_ANCHORS = {
    "always-fold": (-64.6, 3.3),
    "always-call": (-241.1, 26.2),
    "always-all-in": (-380.6, 4.3),
}


class GtoWizardError(RuntimeError):
    """Protocol, mirroring, or transport failure for a single hand."""


def api_key() -> str:
    """The key from the environment, falling back to the gitignored .env."""
    key = os.environ.get("GTOWIZARD_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GTOWIZARD_API_KEY" and value.strip():
                return value.strip()
    raise GtoWizardError(
        "no GTOWIZARD_API_KEY in the environment or .env; request one at "
        "https://benchmark.gtowizard.com/"
    )


_BOT_NAME: str | None = None


def bot_name() -> str:
    """This key's registered bot name, fetched once and cached.

    Used to identify the hero seat unambiguously; see MirroredHand._seats.
    """
    global _BOT_NAME
    if _BOT_NAME is None:
        _BOT_NAME = str(server_results()["bot_name"])
    return _BOT_NAME


def _request(method: str, path: str, payload: dict | None = None,
             params: dict | None = None, *, retries: int = 4,
             timeout: float = 60.0) -> dict:
    url = f"{BASE_URL}{path}"
    if params:
        from urllib.parse import urlencode

        url = f"{url}?{urlencode(params)}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("X-API-Key", api_key())
        # Cloudflare fronts this API and rejects the default "Python-urllib/x.y"
        # signature with HTTP 403 / error code 1010, while the identical request
        # from curl succeeds. An explicit User-Agent is required, not cosmetic.
        request.add_header("User-Agent", USER_AGENT)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            # 4xx other than 429 is our bug; retrying just burns quota.
            if 400 <= error.code < 500 and error.code != 429:
                raise GtoWizardError(f"{method} {path} -> HTTP {error.code}: {detail}") from error
            last = GtoWizardError(f"{method} {path} -> HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
        if attempt < retries - 1:
            time.sleep(2.0 * (attempt + 1))
    raise GtoWizardError(f"{method} {path} failed after {retries} attempts: {last}")


def parse_cards(text: str) -> list[tuple[int, str]]:
    """"9c4d5h" -> [(9,'c'), (4,'d'), (5,'h')]. Every card is exactly two chars."""
    text = (text or "").strip()
    if len(text) % 2:
        raise GtoWizardError(f"odd-length card string {text!r}")
    return [(_RANK_MAP[text[i]], _SUIT_MAP[text[i + 1]])
            for i in range(0, len(text), 2)]


def encode_action(event: dict) -> tuple[str, int | None]:
    """Engine public-action event -> (API action char, amount).

    The engine records a raise with `amount` = the street-cumulative raise-to
    total, which is exactly this API's `b` semantics. All-ins are recorded as a
    raise to the full stack (or a call when short), so the event already states
    what really happened and no special case is needed.
    """
    action = event["action"]
    if action == "fold":
        return "f", None
    if action == "check":
        return "k", None
    if action == "call":
        return "c", None
    if action == "raise":
        return "b", int(event["amount"])
    raise GtoWizardError(f"cannot encode engine action {action!r}")


class MirroredHand:
    """Replays a GTO Wizard hand into a local engine. The hero is seat 0."""

    def __init__(self, response: dict, seed: int = 0) -> None:
        state = response["game_state"]
        hero, villain = self._seats(state)
        self.hero_name = hero["name"]
        self.hero_position = hero["position"]
        # Heads-up, the small blind IS the button and acts first preflop.
        self.hero_is_button = hero["position"].upper() == "SB"
        desired_button = 0 if self.hero_is_button else 1

        # button_offset must be set AT CONSTRUCTION. Re-dealing to move the
        # button posts the blinds a second time out of already blind-reduced
        # stacks, silently corrupting depth and SPR on every out-of-position
        # hand -- a bug that really happened in the Slumbot mirror.
        engine = HeadsUpHoldem(
            initial_stack=STACK,
            small_blind=SMALL_BLIND,
            big_blind=BIG_BLIND,
            rng=random.Random(seed),
            button_offset=desired_button,
        )
        if engine.button != desired_button:
            raise GtoWizardError(f"could not seat the button at {desired_button}")
        if sum(engine.stacks) + engine.pot != 2 * STACK:
            raise GtoWizardError(
                f"blind accounting is wrong: stacks={engine.stacks} pot={engine.pot}")
        self.engine = engine
        self.engine.hole_cards[0] = parse_cards(hero["hole_cards"])
        self._resync_deck()
        self.applied_tokens = 0
        self.board_desyncs = 0
        self.sync(response)

    @staticmethod
    def _seats(state: dict) -> tuple[dict, dict]:
        """Split the seats into (hero, villain).

        Identified by NAME, not by which hole cards are visible. The visible-card
        heuristic excluded 5.6% of hands in the first null run: the villain's
        cards are revealed whenever the hand is already decided, so both seats
        can be visible and the hero becomes ambiguous. The registered bot name is
        unambiguous in every state.
        """
        players = state["players"]
        if len(players) != 2:
            raise GtoWizardError(f"expected two players, got {len(players)}")
        name = bot_name()
        heroes = [p for p in players if p.get("name") == name]
        if len(heroes) == 1:
            hero = heroes[0]
        else:
            # Fall back to visible hole cards only if the name did not resolve.
            visible = [p for p in players if p.get("hole_cards")]
            if len(visible) != 1:
                names = [p.get("name") for p in players]
                raise GtoWizardError(
                    f"could not identify the hero (bot_name={name!r}, seats={names})")
            hero = visible[0]
        villain = next(p for p in players if p is not hero)
        if not hero.get("hole_cards"):
            raise GtoWizardError("hero seat has no visible hole cards")
        return hero, villain

    # -- card bookkeeping ---------------------------------------------------

    def _resync_deck(self) -> None:
        """Keep the engine's phantom cards disjoint from the real ones.

        The engine deals its own opponent hand and board. Both are fiction, but
        they must stay LEGAL fiction: a phantom card duplicating a real hole or
        board card makes equity, bucket lookups and blocker maths nonsense.
        """
        engine = self.engine
        known = set(engine.hole_cards[0]) | set(engine.community)
        available = [card for card in new_deck() if card not in known]
        engine.rng.shuffle(available)
        opponent = [card for card in engine.hole_cards[1] if card not in known]
        while len(opponent) < 2:
            opponent.append(available.pop())
        engine.hole_cards[1] = opponent[:2]
        blocked = known | set(engine.hole_cards[1])
        engine.deck = [card for card in available if card not in blocked]

    def _overwrite_board(self, board: list[tuple[int, str]]) -> None:
        engine = self.engine
        shared = min(len(engine.community), len(board))
        for index in range(shared):
            engine.community[index] = board[index]
        if len(board) > len(engine.community):
            # A street was revealed that the engine has not reached: a mirroring
            # error, not a legal state. Surface it rather than paper over it.
            self.board_desyncs += 1

    # -- replay -------------------------------------------------------------

    def sync(self, response: dict) -> None:
        """Apply the villain's unseen action tokens and the current board."""
        state = response["game_state"]
        board = parse_cards(state.get("board_cards", ""))
        # "_" is an end-of-round marker; the engine advances streets itself.
        tokens = [t for t in state.get("action_history", []) if t != "_"]
        for token in tokens[self.applied_tokens:]:
            self._apply(token, board)
        self.applied_tokens = len(tokens)
        self._overwrite_board(board)
        self._resync_deck()

    def _apply(self, token: str, board: list[tuple[int, str]]) -> None:
        engine = self.engine
        player = engine.current_player
        if player is None:
            raise GtoWizardError(f"action token {token!r} after the hand completed")
        if token == "f":
            engine.act(player, "fold")
        elif token == "k":
            engine.act(player, "check")
        elif token == "c":
            engine.act(player, "call")
        elif token.startswith("b"):
            target = int(round(float(token[1:])))
            legal = engine.legal_actions(player)
            if not legal:
                raise GtoWizardError(f"bet token {token!r} but seat {player} cannot act")
            if target >= int(legal["raise_max"]):
                engine.act(player, "all_in")
            else:
                engine.act(player, "raise", max(target, int(legal["raise_min"])))
        else:
            raise GtoWizardError(f"unknown action token {token!r}")
        self._overwrite_board(board)
        self._resync_deck()

    def act_for_hero(self, agent) -> tuple[str, int | None]:
        """Let the agent choose and apply its own action; return the API token."""
        engine = self.engine
        if engine.hand_complete:
            raise GtoWizardError("asked for a hero action after the hand completed")
        if engine.current_player != 0:
            raise GtoWizardError(
                f"hero asked to act out of turn (current={engine.current_player})")
        before = len(engine.public_actions)
        choice = agent.select(engine, 0)
        agent.execute(engine, 0, choice)
        if len(engine.public_actions) <= before:
            raise GtoWizardError("agent.execute did not produce an engine action")
        event = dict(engine.public_actions[-1])
        if int(event["player"]) != 0:
            raise GtoWizardError("agent acted for the wrong seat")
        # Our own action reappears in the next response's history; count it as
        # applied so sync() does not replay it.
        self.applied_tokens += 1
        return encode_action(event)


def _summarize(samples: list[float], *, per_bb: float) -> dict:
    if not samples:
        return {"bb_per_100": 0.0, "ci_low": 0.0, "ci_high": 0.0, "hands": 0, "stdev": 0.0}
    in_bb = [value / per_bb for value in samples]
    mean = statistics.fmean(in_bb)
    deviation = statistics.stdev(in_bb) if len(in_bb) > 1 else 0.0
    margin = 1.96 * deviation / (len(in_bb) ** 0.5)
    return {
        "bb_per_100": round(mean * 100, 2),
        "ci_low": round((mean - margin) * 100, 2),
        "ci_high": round((mean + margin) * 100, 2),
        "stdev_bb": round(deviation, 4),
        "hands": len(in_bb),
    }


@dataclass
class MatchState:
    """Resumable progress. Flushed after every hand, so a crash costs one hand."""

    hands_played: int = 0
    winnings: list[float] = field(default_factory=list)
    aivat: list[float] = field(default_factory=list)
    excluded: int = 0
    board_desyncs: int = 0
    positions: dict = field(default_factory=dict)
    timing: dict = field(default_factory=lambda: {"api_s": 0.0, "agent_s": 0.0, "mirror_s": 0.0})

    def to_json(self) -> dict:
        return {
            "hands_played": self.hands_played,
            "winnings": self.winnings,
            "aivat": self.aivat,
            "excluded": self.excluded,
            "board_desyncs": self.board_desyncs,
            "positions": self.positions,
            "timing": {k: round(v, 1) for k, v in self.timing.items()},
        }

    @classmethod
    def from_json(cls, payload: dict) -> "MatchState":
        state = cls()
        state.hands_played = int(payload.get("hands_played", 0))
        state.winnings = list(payload.get("winnings", []))
        state.aivat = list(payload.get("aivat", []))
        state.excluded = int(payload.get("excluded", 0))
        state.board_desyncs = int(payload.get("board_desyncs", 0))
        state.positions = dict(payload.get("positions", {}))
        state.timing = dict(payload.get("timing", {"api_s": 0.0, "agent_s": 0.0, "mirror_s": 0.0}))
        return state


def play_match(agent, hands: int, *, seed: int = 0, log=print,
               checkpoint: Path | None = None, flush_every: int = 25) -> dict:
    """Play `hands` hands against GTO Wizard AI and report both raw and AIVAT."""

    state = MatchState()
    if checkpoint and checkpoint.exists():
        state = MatchState.from_json(json.loads(checkpoint.read_text(encoding="utf-8")))
        log(f"resuming at {state.hands_played} hands ({state.excluded} excluded)")

    def flush() -> None:
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(state.to_json(), indent=2), encoding="utf-8")

    started = time.monotonic()
    while state.hands_played + state.excluded < hands:
        index = state.hands_played + state.excluded
        mirror = None
        try:
            mark = time.monotonic()
            response = _request("POST", "/hands", {"game_name": GAME_NAME})
            state.timing["api_s"] += time.monotonic() - mark

            mark = time.monotonic()
            mirror = MirroredHand(response, seed=seed * 1_000_003 + index)
            state.timing["mirror_s"] += time.monotonic() - mark
            hand_id = int(response["hand_id"])

            while not response["game_state"]["is_hand_over"]:
                if mirror.engine.hand_complete:
                    raise GtoWizardError(
                        "mirror completed the hand before the API reported it over")
                mark = time.monotonic()
                action, amount = mirror.act_for_hero(agent)
                state.timing["agent_s"] += time.monotonic() - mark

                payload: dict = {"action": action}
                if amount is not None:
                    payload["amount"] = amount
                mark = time.monotonic()
                response = _request("POST", f"/hands/{hand_id}/act", payload)
                state.timing["api_s"] += time.monotonic() - mark

                mark = time.monotonic()
                mirror.sync(response)
                state.timing["mirror_s"] += time.monotonic() - mark

            final = response["game_state"]
            if final.get("winnings") is None or final.get("aivat_score") is None:
                raise GtoWizardError("hand over but winnings/aivat_score missing")
            state.winnings.append(float(final["winnings"]))
            state.aivat.append(float(final["aivat_score"]))
            state.hands_played += 1
            state.board_desyncs += mirror.board_desyncs
            key = f"hero={mirror.hero_position}"
            state.positions[key] = state.positions.get(key, 0) + 1

        except GtoWizardError as error:
            # Excluded, never folded-and-scored: folding our own mirroring bug
            # would book a real loss caused by us.
            state.excluded += 1
            log(f"  hand {index} EXCLUDED: {error}")

        done = state.hands_played + state.excluded
        if done % flush_every == 0 or done == hands:
            flush()
            elapsed = time.monotonic() - started
            rate = state.hands_played / max(elapsed, 1e-9)
            remaining = (hands - done) / max(rate, 1e-9)
            raw = _summarize(state.winnings, per_bb=BIG_BLIND)
            adj = _summarize(state.aivat, per_bb=BIG_BLIND)
            log(f"  {done}/{hands} hands | AIVAT {adj['bb_per_100']:+.2f} "
                f"[{adj['ci_low']:+.2f},{adj['ci_high']:+.2f}] | raw "
                f"{raw['bb_per_100']:+.2f} | {rate:.2f} hands/s | ETA {remaining/60:.1f} min")

    flush()
    report = {
        "instrument": "gto-wizard-ai",
        "game_name": GAME_NAME,
        "stack_bb": STACK / BIG_BLIND,
        "hands_requested": hands,
        "hands_scored": state.hands_played,
        "excluded": state.excluded,
        "board_desyncs": state.board_desyncs,
        "positions": state.positions,
        "aivat": _summarize(state.aivat, per_bb=BIG_BLIND),
        "raw": _summarize(state.winnings, per_bb=BIG_BLIND),
        "timing_s": {k: round(v, 1) for k, v in state.timing.items()},
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    return report


def server_results() -> dict:
    """The API's own tally, for cross-checking our local arithmetic."""
    return _request("GET", "/results", params={"game_name": GAME_NAME})
