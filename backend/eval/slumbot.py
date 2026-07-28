"""Slumbot API harness: the project's only external absolute strength measure.

Slumbot plays 200bb heads-up NLHE (20,000 chips, 50/100 blinds), stacks reset
every hand. Protocol (slumbot.com/api): POST new_hand -> {token, client_pos,
hole_cards, action, board}; POST act {token, incr} where incr is one of
'f' (fold), 'k' (check), 'c' (call), 'b<chips>' (raise to a STREET-CUMULATIVE
total). '/' separates streets in the returned action string.

Design rules this harness follows, learned the hard way (docs/STATUS.md §4):

1. **The agent's own action mapping is used, never a reimplementation.** We call
   `agent.select()` then `agent.execute()` — exactly what `duel.py` and live
   serving do — and then *derive* the Slumbot token from the engine event the
   agent produced. The previous version hand-rolled a 4-way mapping with a
   hard-coded 0.5-pot raise, which measured a crippled agent regardless of the
   blueprint's real sizing.
2. **Position comes from the protocol, not from a constant.** On a `new_hand`
   response an empty action string means nobody has acted yet, so the client is
   first to act, which in heads-up is the button/small blind. Public
   descriptions of `client_pos` contradict each other, so the observed
   client_pos -> seat mapping is *reported* rather than assumed.
3. **Broken hands are excluded, not folded and counted.** A mirror error used
   to fold the hand and still bank the winnings, silently biasing the mean.
   Now such hands are dropped from the sample and counted; a high error rate
   invalidates the report.
4. **The engine's deck is sanitized** against the real cards, so its phantom
   deals (the opponent's hole cards, unrevealed board) can never duplicate a
   card Slumbot actually dealt.
5. **NULL-testable offline.** `tests/test_slumbot_harness.py` drives this module
   against a local fake server that speaks the same protocol, and asserts the
   analytic always-fold result. No number from here is believed until that
   passes.

CLI:
    python -m backend.eval.slumbot --hands 20000 --gpu --log runs/slumbot.jsonl
    python -m backend.eval.slumbot --hands 200 --null always-fold   # harness check
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from backend.poker import HeadsUpHoldem, new_deck

HOST = "https://slumbot.com"
STACK = 20_000
SMALL_BLIND, BIG_BLIND = 50, 100

_SUIT_MAP = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
_RANK_MAP = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
_SUIT_INVERSE = {glyph: letter for letter, glyph in _SUIT_MAP.items()}
_RANK_INVERSE = {rank: letter for letter, rank in _RANK_MAP.items()}


class SlumbotError(RuntimeError):
    """Protocol, mirroring, or transport failure for a single hand."""


# -- transport ------------------------------------------------------------------


def _post(path: str, payload: dict, retries: int = 4, timeout: float = 30.0) -> dict:
    """POST with backoff. Transport faults must not end a 20,000-hand session."""
    delay = 1.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{HOST}/api/{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
            if attempt + 1 < retries:
                time.sleep(delay)
                delay *= 2
    raise SlumbotError(f"POST {path} failed after {retries} attempts: {last}")


# -- protocol encoding ----------------------------------------------------------


def parse_card(text: str) -> tuple[int, str]:
    return _RANK_MAP[text[0]], _SUIT_MAP[text[1]]


def format_card(card: tuple[int, str]) -> str:
    return f"{_RANK_INVERSE[card[0]]}{_SUIT_INVERSE[card[1]]}"


def tokenize(action_string: str) -> list[str]:
    """Split a Slumbot action string into one token per player action."""
    tokens: list[str] = []
    for street_actions in action_string.split("/"):
        index = 0
        while index < len(street_actions):
            char = street_actions[index]
            if char in ("k", "c", "f"):
                tokens.append(char)
                index += 1
            elif char == "b":
                end = index + 1
                while end < len(street_actions) and street_actions[end].isdigit():
                    end += 1
                if end == index + 1:
                    raise SlumbotError(f"bet token without an amount in {action_string!r}")
                tokens.append(street_actions[index:end])
                index = end
            else:
                raise SlumbotError(f"unknown action token at {street_actions[index:]!r}")
    return tokens


def encode_action(event: dict) -> str:
    """Slumbot token for an engine public-action event.

    The engine records a raise with `amount` = the street-cumulative raise-to
    total, which is exactly Slumbot's `b<total>` semantics. All-ins are recorded
    as a raise to the full stack (or a call when the stack is short), so no
    special case is needed here — the event always states what really happened.
    """
    action = event["action"]
    if action == "fold":
        return "f"
    if action == "check":
        return "k"
    if action == "call":
        return "c"
    if action == "raise":
        return f"b{int(event['amount'])}"
    raise SlumbotError(f"cannot encode engine action {action!r} for Slumbot")


# -- hand mirroring -------------------------------------------------------------


class MirroredHand:
    """Replays a Slumbot hand into a local engine. The client is seat 0."""

    def __init__(self, response: dict, seed: int = 0) -> None:
        self.client_pos = int(response["client_pos"])
        # Protocol-derived position: nobody has acted yet iff we are first to
        # act, and in heads-up the button/small blind acts first preflop.
        self.client_is_button = not str(response.get("action", ""))
        desired_button = 0 if self.client_is_button else 1
        # button_offset must be set AT CONSTRUCTION. Re-dealing with new_hand()
        # to move the button posts the blinds a second time out of already
        # blind-reduced stacks (19,850 instead of 19,950/19,900), which silently
        # corrupts stack depth and SPR for every out-of-position hand. On hand 1
        # the engine's button is simply the offset.
        engine = HeadsUpHoldem(
            initial_stack=STACK,
            small_blind=SMALL_BLIND,
            big_blind=BIG_BLIND,
            rng=random.Random(seed),
            button_offset=desired_button,
        )
        if engine.button != desired_button:  # pragma: no cover - offset is exact on hand 1
            raise SlumbotError(f"could not seat the button at {desired_button}")
        if sum(engine.stacks) + engine.pot != 2 * STACK:
            raise SlumbotError(f"blind accounting is wrong: stacks={engine.stacks} pot={engine.pot}")
        self.engine = engine
        self.engine.hole_cards[0] = [parse_card(card) for card in response["hole_cards"]]
        self._resync_deck()
        self.applied_tokens = 0
        self.board_desyncs = 0

    # -- card bookkeeping ---------------------------------------------------

    def _resync_deck(self) -> None:
        """Keep the engine's unknown cards disjoint from Slumbot's real ones.

        The engine deals its own opponent hand and its own board. Both are
        fiction here, but they must remain *legal* fiction: a phantom card that
        duplicates one of our real hole cards or a real board card would make
        equity/bucket lookups and any search blocker math nonsense.
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
        """Replace the engine's phantom board with the real one, as far as both exist."""
        engine = self.engine
        shared = min(len(engine.community), len(board))
        for index in range(shared):
            engine.community[index] = board[index]
        if len(board) > len(engine.community):
            # Slumbot revealed a street the engine has not reached: a mirroring
            # error, not a legal state. Surface it rather than papering over it.
            self.board_desyncs += 1

    # -- replay -------------------------------------------------------------

    def sync(self, response: dict) -> None:
        """Apply the opponent's unseen action tokens and the current board."""
        board = [parse_card(card) for card in response.get("board", [])]
        tokens = tokenize(str(response.get("action", "")))
        for token in tokens[self.applied_tokens :]:
            self._apply(token, board)
        self.applied_tokens = len(tokens)
        self._overwrite_board(board)
        self._resync_deck()

    def _apply(self, token: str, board: list[tuple[int, str]]) -> None:
        engine = self.engine
        player = engine.current_player
        if player is None:
            raise SlumbotError(f"action token {token!r} after the hand completed")
        if token == "f":
            engine.act(player, "fold")
        elif token == "k":
            engine.act(player, "check")
        elif token == "c":
            engine.act(player, "call")
        else:
            target = int(token[1:])
            legal = engine.legal_actions(player)
            if not legal:
                raise SlumbotError(f"bet token {token!r} but seat {player} cannot act")
            if target >= int(legal["raise_max"]):
                engine.act(player, "all_in")
            else:
                engine.act(player, "raise", max(target, int(legal["raise_min"])))
        # Correct the board as early as possible: a street may have just
        # advanced and dealt phantom cards.
        self._overwrite_board(board)
        self._resync_deck()

    def act_for_client(self, agent) -> tuple[str, dict]:
        """Let the agent choose and apply its own action; return the token."""
        engine = self.engine
        if engine.hand_complete:
            raise SlumbotError("asked for a client action after the hand completed")
        if engine.current_player != 0:
            raise SlumbotError(f"client asked to act out of turn (current={engine.current_player})")
        before = len(engine.public_actions)
        choice = agent.select(engine, 0)
        agent.execute(engine, 0, choice)
        if len(engine.public_actions) <= before:
            raise SlumbotError("agent.execute did not produce an engine action")
        event = dict(engine.public_actions[-1])
        if int(event["player"]) != 0:
            raise SlumbotError("agent acted for the wrong seat")
        # Our own action reappears in the next response's action string; count
        # it as applied so sync() does not replay it.
        self.applied_tokens += 1
        return encode_action(event), event


# -- match --------------------------------------------------------------------


def _summarize(samples: list[float]) -> dict:
    if not samples:
        return {"hands": 0, "bb_per_100": None, "ci_low": None, "ci_high": None}
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(samples))
    return {
        "hands": len(samples),
        "bb_per_100": round(mean * 100, 2),
        "ci_low": round((mean - margin) * 100, 2),
        "ci_high": round((mean + margin) * 100, 2),
        "stdev_bb": round(deviation, 4),
    }


def play_match(
    agent,
    hands: int = 100,
    token: str | None = None,
    progress: bool = True,
    aivat: bool = True,
    log_path: Path | None = None,
    post=_post,
    seed: int = 0,
) -> dict:
    """Play `hands` hands and report bb/100 with CIs, positions and errors.

    `post` is injectable so the offline NULL test can drive a local fake server
    through exactly this code path.
    """
    corrector_factory = None
    if aivat:
        from backend.eval.aivat import ChanceCorrector

        corrector_factory = ChanceCorrector

    raw: list[float] = []
    corrected: list[float] = []
    by_position: dict[str, list[float]] = {"button": [], "big_blind": []}
    position_map: dict[str, int] = {}
    errors: dict[str, int] = {}
    excluded = 0
    desyncs = 0
    log = log_path.open("a", encoding="utf-8") if log_path is not None else None
    started = time.monotonic()
    # Wall-clock attribution. A 20,000-hand session is a 12-35 hour commitment,
    # so where the time goes has to be measured rather than assumed (an earlier
    # estimate left ~4.6s/hand of the observed 8.5s unaccounted for).
    timing = {"api_s": 0.0, "agent_s": 0.0, "aivat_s": 0.0, "mirror_s": 0.0}
    api_calls = 0

    def timed_post(path: str, payload: dict) -> dict:
        nonlocal api_calls
        mark = time.monotonic()
        try:
            return post(path, payload)
        finally:
            timing["api_s"] += time.monotonic() - mark
            api_calls += 1

    try:
        for hand_index in range(hands):
            response = timed_post("new_hand", {"token": token} if token else {})
            token = response.get("token", token)
            mirror = None
            corrector = None
            decisions: list[dict] = []
            try:
                mark = time.monotonic()
                mirror = MirroredHand(response, seed=seed * 1_000_003 + hand_index)
                timing["mirror_s"] += time.monotonic() - mark
                if corrector_factory is not None:
                    corrector = corrector_factory(mirror.engine, seat=0, seed=hand_index)
                while response.get("winnings") is None:
                    mark = time.monotonic()
                    mirror.sync(response)
                    timing["mirror_s"] += time.monotonic() - mark
                    if corrector is not None:
                        mark = time.monotonic()
                        corrector.observe(mirror.engine)
                        timing["aivat_s"] += time.monotonic() - mark
                    if mirror.engine.hand_complete:
                        # Slumbot has not reported winnings yet but our mirror
                        # thinks the hand is over: a mirroring divergence.
                        raise SlumbotError("mirror completed the hand before Slumbot reported winnings")
                    mark = time.monotonic()
                    incr, event = mirror.act_for_client(agent)
                    timing["agent_s"] += time.monotonic() - mark
                    decisions.append(
                        {
                            "street": int(event["street"]),
                            "action": str(event["action"]),
                            "amount": int(event["amount"]),
                            "pot_before": int(event.get("pot_before", 0)),
                            "to_call_before": int(event.get("to_call_before", 0)),
                            "incr": incr,
                        }
                    )
                    response = timed_post("act", {"token": token, "incr": incr})
                    token = response.get("token", token)
                    if "error_msg" in response:
                        raise SlumbotError(f"slumbot rejected {incr!r}: {response['error_msg']}")
                mark = time.monotonic()
                mirror.sync(response)
                timing["mirror_s"] += time.monotonic() - mark
                if corrector is not None:
                    mark = time.monotonic()
                    corrector.observe(mirror.engine)
                    timing["aivat_s"] += time.monotonic() - mark
            except (SlumbotError, Exception) as error:  # noqa: BLE001 - one hand must not kill the session
                if isinstance(error, KeyboardInterrupt):
                    raise
                label = type(error).__name__ + ": " + str(error)[:120]
                errors[label] = errors.get(label, 0) + 1
                excluded += 1
                if progress:
                    print(f"hand {hand_index}: EXCLUDED ({label})")
                # Concede the hand so the session can continue, but do NOT
                # count its winnings: a folded-out hand is not a measurement.
                try:
                    if response.get("winnings") is None:
                        timed_post("act", {"token": token, "incr": "f"})
                except Exception:
                    pass
                if log is not None:
                    log.write(json.dumps({"hand": hand_index, "excluded": True, "error": label}) + "\n")
                continue

            winnings = response.get("winnings")
            if winnings is None:
                excluded += 1
                continue
            result_bb = float(winnings) / BIG_BLIND
            raw.append(result_bb)
            position = "button" if mirror.client_is_button else "big_blind"
            by_position[position].append(result_bb)
            key = f"client_pos={mirror.client_pos}->{position}"
            position_map[key] = position_map.get(key, 0) + 1
            desyncs += mirror.board_desyncs
            record: dict = {
                "hand": hand_index,
                "position": position,
                "client_pos": mirror.client_pos,
                "winnings_bb": round(result_bb, 4),
                "hole_cards": response.get("hole_cards"),
                "board": response.get("board"),
                "action": response.get("action"),
                "decisions": decisions,
            }
            if corrector is not None:
                value = result_bb - corrector.total_bb(BIG_BLIND)
                corrected.append(value)
                record["aivat_bb"] = round(value, 4)
            if log is not None:
                log.write(json.dumps(record) + "\n")
                log.flush()

            if progress and (hand_index + 1) % 100 == 0:
                rate = (time.monotonic() - started) / (hand_index + 1)
                line = f"{hand_index + 1}/{hands} hands: {_summarize(raw)['bb_per_100']:+.1f} bb/100"
                if corrected:
                    line += f" | AIVAT {_summarize(corrected)['bb_per_100']:+.1f}"
                if excluded:
                    line += f" | excluded {excluded}"
                line += f" | {rate:.1f}s/hand"
                # flush: stdout is block-buffered when redirected to a file, so a
                # multi-hour run would otherwise show no progress at all.
                print(line, flush=True)
    finally:
        if log is not None:
            log.close()

    report = {
        "hands": len(raw),
        "hands_attempted": hands,
        "excluded": excluded,
        "error_rate": round(excluded / max(hands, 1), 6),
        "errors": errors,
        "board_desyncs": desyncs,
        "elapsed_s": round(time.monotonic() - started, 1),
        "timing": {
            **{key: round(value, 1) for key, value in timing.items()},
            "api_calls": api_calls,
            "unaccounted_s": round(
                time.monotonic() - started - sum(timing.values()),
                1,
            ),
            "s_per_hand": round((time.monotonic() - started) / max(hands, 1), 2),
        },
        "token": token,
        "raw": _summarize(raw),
        "by_position": {name: _summarize(values) for name, values in by_position.items()},
        "client_pos_mapping": position_map,
    }
    report["bb_per_100"] = report["raw"]["bb_per_100"]
    report["ci_low"] = report["raw"]["ci_low"]
    report["ci_high"] = report["raw"]["ci_high"]
    if corrected:
        report["aivat"] = _summarize(corrected)
        raw_dev = report["raw"].get("stdev_bb") or 0.0
        aivat_dev = report["aivat"].get("stdev_bb") or 0.0
        if raw_dev > 0:
            report["aivat_variance_reduction"] = round(1.0 - (aivat_dev / raw_dev) ** 2, 3)
    return report


def load_agent(use_gpu: bool, null_policy: str | None, subgame_iters: int) -> object:
    if null_policy:
        from backend.eval.null_agents import ScriptedAgent

        return ScriptedAgent(null_policy)
    if use_gpu:
        from backend.agents.multistack_agent import MultiStackBlueprintAgent

        agent = MultiStackBlueprintAgent.try_load()
        if agent is None:
            from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

            agent = GpuBlueprintAgent.try_load()
        if agent is not None:
            # Search stays off unless explicitly requested: the serving default
            # is blueprint-only (docs/STATUS.md §1).
            for target in (agent, *getattr(agent, "agents", {}).values()):
                if hasattr(target, "subgame_search"):
                    target.subgame_search = subgame_iters > 0
                    if subgame_iters > 0:
                        target.subgame_iterations = subgame_iters
        return agent
    from backend.agents.blueprint_agent import BlueprintAgent

    return BlueprintAgent.try_load()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the serving agent against Slumbot")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--gpu", action="store_true", help="serve the GPU blueprint champion")
    parser.add_argument("--subgame-iters", type=int, default=0, help=">0 enables live re-solving (default off)")
    parser.add_argument("--null", type=str, default=None, help="run a scripted NULL agent instead (e.g. always-fold)")
    parser.add_argument("--no-aivat", action="store_true")
    parser.add_argument("--log", type=str, default=None, help="append per-hand JSONL here")
    parser.add_argument("--token", type=str, default=None, help="resume an existing Slumbot session token")
    parser.add_argument("--report", type=str, default=None, help="write the final JSON report here")
    arguments = parser.parse_args()

    agent = load_agent(arguments.gpu, arguments.null, arguments.subgame_iters)
    if agent is None:
        raise SystemExit("no agent artifacts found")

    log_path = Path(arguments.log) if arguments.log else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    report = play_match(
        agent,
        hands=arguments.hands,
        token=arguments.token,
        aivat=not arguments.no_aivat,
        log_path=log_path,
    )

    raw = report["raw"]
    print(
        f"\nSlumbot: {raw['bb_per_100']:+.2f} bb/100 "
        f"[{raw['ci_low']:+.2f}, {raw['ci_high']:+.2f}] over {report['hands']} hands"
    )
    if "aivat" in report:
        aivat = report["aivat"]
        print(
            f"AIVAT:   {aivat['bb_per_100']:+.2f} bb/100 "
            f"[{aivat['ci_low']:+.2f}, {aivat['ci_high']:+.2f}] "
            f"(variance -{report.get('aivat_variance_reduction', 0) * 100:.0f}%)"
        )
    for name, summary in report["by_position"].items():
        if summary["hands"]:
            print(f"  {name:>10}: {summary['bb_per_100']:+.2f} bb/100 over {summary['hands']} hands")
    print(f"  client_pos mapping: {report['client_pos_mapping']}")
    if report["excluded"]:
        print(f"  EXCLUDED {report['excluded']} hands ({report['error_rate'] * 100:.2f}%): {report['errors']}")
    if report["board_desyncs"]:
        print(f"  WARNING: {report['board_desyncs']} board desyncs — mirroring is not trustworthy")

    if arguments.report:
        path = Path(arguments.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {path}")


if __name__ == "__main__":
    main()
