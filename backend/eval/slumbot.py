"""Slumbot API harness: play our agent against slumbot.com and report bb/100.

Slumbot plays 200bb heads-up NLHE (20,000 chips, 50/100 blinds), stacks reset
every hand. Protocol (slumbot.com/api): POST new_hand -> {token, client_pos,
hole_cards, action, board}; POST act {token, incr} where incr is one of
'f', 'c', 'k', 'b<chips>' (street-cumulative bet-to amount). client_pos 0
means we are the BIG BLIND (Slumbot acts first preflop).

The hand is mirrored into a HeadsUpHoldem engine so any serving agent
(BlueprintAgent / GpuBlueprintAgent) can act through its normal contract.
NOTE: our blueprints train at 100bb — playing 200bb is a known depth
mismatch until multi-stack blueprints land; results are still comparable
run-to-run.

CLI:  python -m backend.eval.slumbot --hands 200 [--gpu]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.request

from backend.poker import SUITS, HeadsUpHoldem
from backend.rl_env import continuous_raise_target

HOST = "https://slumbot.com"
STACK = 20_000
SMALL_BLIND, BIG_BLIND = 50, 100

_SUIT_MAP = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
_RANK_MAP = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _api(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{HOST}/api/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_card(text: str) -> tuple[int, str]:
    return _RANK_MAP[text[0]], _SUIT_MAP[text[1]]


def _tokenize(street_actions: str) -> list[str]:
    tokens: list[str] = []
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
            tokens.append(street_actions[index:end])
            index = end
        else:
            raise ValueError(f"unknown action token at '{street_actions[index:]}'")
    return tokens


class MirroredHand:
    """Replays Slumbot's action string into a local engine; client is seat 0."""

    def __init__(self, response: dict) -> None:
        self.client_pos = int(response["client_pos"])  # 0 -> we are the big blind
        engine = HeadsUpHoldem(initial_stack=STACK, small_blind=SMALL_BLIND, big_blind=BIG_BLIND)
        # Seat 0 = client. The engine's button posts the small blind and acts
        # first preflop; Slumbot's client_pos 0 means the client is BB.
        desired_button = 1 if self.client_pos == 0 else 0
        if engine.button != desired_button:
            # new_hand() increments hand_number, so pick the offset that lands
            # the button on the desired seat after that increment.
            engine.button_offset = (desired_button - engine.hand_number) % 2
            engine.new_hand()
        self.engine = engine
        self.engine.hole_cards[0] = [_parse_card(card) for card in response["hole_cards"]]
        self.applied_tokens = 0

    def sync(self, response: dict) -> None:
        """Apply unseen action tokens and the current board to the engine."""
        board = [_parse_card(card) for card in response.get("board", [])]
        action = response.get("action", "")
        tokens: list[str] = []
        for street_actions in action.split("/"):
            tokens.extend(_tokenize(street_actions))
        for token in tokens[self.applied_tokens :]:
            self._apply(token, board)
        self.applied_tokens = len(tokens)
        self._overwrite_board(board)

    def _overwrite_board(self, board: list[tuple[int, str]]) -> None:
        # The engine deals its own random cards on street changes; replace
        # them with the real board wherever both exist.
        game = self.engine
        for index in range(min(len(game.community), len(board))):
            game.community[index] = board[index]

    def _apply(self, token: str, board: list[tuple[int, str]]) -> None:
        game = self.engine
        player = game.current_player
        if player is None:
            raise ValueError("action token after hand completion")
        if token == "f":
            game.act(player, "fold")
        elif token == "k":
            game.act(player, "check")
        elif token == "c":
            game.act(player, "call")
        else:
            target = int(token[1:])
            legal = game.legal_actions(player)
            if target >= int(legal.get("raise_max", target)):
                game.act(player, "all_in")
            else:
                game.act(player, "raise", max(target, int(legal.get("raise_min", target))))
        self._overwrite_board(board)

    def client_incr(self, agent) -> str:
        """Ask the agent for the client's action; apply locally and encode."""
        game = self.engine
        legal = game.legal_actions(0)
        # Our own action will appear in the next response's action string;
        # count it as applied so sync() does not replay it.
        self.applied_tokens += 1
        choice = agent.select(game, 0)
        raise_fraction = getattr(agent, "_raise_fraction", None)
        if choice == 0 and legal.get("fold"):
            game.act(0, "fold")
            return "f"
        if choice == 3 and legal.get("all_in"):
            target = int(legal["raise_max"])
            game.act(0, "all_in")
            return f"b{target}"
        if choice == 2 and legal.get("raise"):
            target = continuous_raise_target(game, 0, 0.5 if raise_fraction is None else raise_fraction)
            game.act(0, "raise", target)
            return f"b{target}"
        if legal.get("check"):
            game.act(0, "check")
            return "k"
        game.act(0, "call")
        return "c"


def play_match(agent, hands: int = 100, token: str | None = None, progress: bool = True) -> dict:
    results_bb: list[float] = []
    for hand_index in range(hands):
        response = _api("new_hand", {"token": token} if token else {})
        token = response.get("token", token)
        try:
            mirror = MirroredHand(response)
            while "winnings" not in response or response["winnings"] is None:
                mirror.sync(response)
                if mirror.engine.hand_complete:
                    break
                incr = mirror.client_incr(agent)
                response = _api("act", {"token": token, "incr": incr})
                token = response.get("token", token)
                if "error_msg" in response:
                    raise ValueError(response["error_msg"])
        except Exception as error:  # noqa: BLE001 - a broken hand must not kill the match
            if progress:
                print(f"hand {hand_index}: mirror error ({error}); folding out")
            try:
                response = _api("act", {"token": token, "incr": "f"})
            except Exception:
                continue
        winnings = response.get("winnings")
        if winnings is not None:
            results_bb.append(winnings / BIG_BLIND)
        if progress and (hand_index + 1) % 25 == 0:
            mean = statistics.fmean(results_bb) if results_bb else 0.0
            print(f"{hand_index + 1} hands: {mean * 100:+.1f} bb/100")

    mean = statistics.fmean(results_bb) if results_bb else 0.0
    deviation = statistics.stdev(results_bb) if len(results_bb) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(max(len(results_bb), 1))
    return {
        "hands": len(results_bb),
        "bb_per_100": round(mean * 100, 2),
        "ci_low": round((mean - margin) * 100, 2),
        "ci_high": round((mean + margin) * 100, 2),
        "token": token,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the serving agent against Slumbot")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--subgame-iters", type=int, default=120, help="0 disables turn/river re-solving")
    arguments = parser.parse_args()
    if arguments.gpu:
        from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

        agent = GpuBlueprintAgent.try_load()
        if agent is not None:
            agent.subgame_search = arguments.subgame_iters > 0
            agent.subgame_iterations = arguments.subgame_iters or agent.subgame_iterations
    else:
        from backend.agents.blueprint_agent import BlueprintAgent

        agent = BlueprintAgent.try_load()
    if agent is None:
        raise SystemExit("no blueprint artifacts found")
    report = play_match(agent, hands=arguments.hands)
    print(
        f"\nSlumbot match: {report['bb_per_100']:+.2f} bb/100 "
        f"[{report['ci_low']:+.2f}, {report['ci_high']:+.2f}] over {report['hands']} hands"
    )


if __name__ == "__main__":
    main()
