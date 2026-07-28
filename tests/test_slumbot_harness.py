"""Slumbot-harness NULL tests, driven by a local fake server.

The harness is the project's only external absolute strength measure, so it is
worthless unless it can first reproduce a number that is known independently.
Three separate eval harnesses have already shipped plausible-looking wrong
numbers (docs/STATUS.md §4), so the rule is: no number is believed until the
instrument reproduces an analytic result.

The anchor here is exact, not statistical: a player that folds at its first
opportunity from the button posts the small blind and loses it, every hand, with
zero variance — exactly -50 bb/100. That single assertion covers the blind
geometry, the seat mapping, the sign convention and the bb normalization, which
is precisely the bug class behind the retracted +75 bb/100 inflation.

`FakeSlumbot` speaks the documented wire protocol over a real rules engine, so
`play_match` runs its true code path with only the transport swapped.
"""

from __future__ import annotations

import random
import unittest

from backend.eval.null_agents import FOLD_OUT_OF_POSITION_BOUNDS, ScriptedAgent
from backend.eval.slumbot import (
    BIG_BLIND,
    SMALL_BLIND,
    STACK,
    MirroredHand,
    SlumbotError,
    encode_action,
    format_card,
    play_match,
    tokenize,
)
from backend.poker import HeadsUpHoldem


def _action_string(engine: HeadsUpHoldem) -> str:
    """Rebuild Slumbot's action string from the engine's public history.

    Street separators appear between streets that have actions, and a trailing
    separator appears when a street has just begun — matching the real API (an
    example real response is `cb300c/kk/kb300c/kb1200c`).
    """
    parts: list[str] = []
    street = 0
    for event in engine.public_actions:
        if event["action"] == "blind":
            continue
        while int(event["street"]) > street:
            parts.append("/")
            street += 1
        parts.append(encode_action(event))
    while engine.street > street:
        parts.append("/")
        street += 1
    return "".join(parts)


class FakeSlumbot:
    """A local stand-in for slumbot.com/api over a real HeadsUpHoldem engine.

    Seat 0 is the client, seat 1 is "Slumbot". `invert_client_pos` flips the
    reported client_pos convention so tests can prove the harness derives
    position from the protocol rather than trusting that field.
    """

    def __init__(self, seed: int = 0, invert_client_pos: bool = False, policy: str = "mixed") -> None:
        self.rng = random.Random(seed)
        self.invert_client_pos = invert_client_pos
        self.policy = policy
        self.hand_index = 0
        self.engine: HeadsUpHoldem | None = None
        self.calls = 0

    # -- transport-shaped entry point --------------------------------------

    def __call__(self, path: str, payload: dict, **kwargs) -> dict:
        self.calls += 1
        if path == "new_hand":
            return self._new_hand()
        if path == "act":
            return self._act(str(payload["incr"]))
        raise AssertionError(f"unexpected path {path!r}")

    # -- protocol ----------------------------------------------------------

    def _new_hand(self) -> dict:
        self.hand_index += 1
        # Alternate the button so both client positions are exercised. The
        # offset goes in at construction: re-dealing to move it would post the
        # blinds twice (the bug this suite caught in the harness itself).
        desired_button = self.hand_index % 2
        engine = HeadsUpHoldem(
            initial_stack=STACK,
            small_blind=SMALL_BLIND,
            big_blind=BIG_BLIND,
            rng=random.Random(self.hand_index * 7919),
            button_offset=desired_button,
        )
        assert engine.button == desired_button
        assert sum(engine.stacks) + engine.pot == 2 * STACK, "fake server blind accounting"
        self.engine = engine
        self._play_opponent()
        return self._response()

    def _act(self, incr: str) -> dict:
        engine = self.engine
        assert engine is not None
        if engine.hand_complete:
            return {**self._response(), "error_msg": "hand already complete"}
        if engine.current_player != 0:
            return {**self._response(), "error_msg": "client acted out of turn"}
        try:
            self._apply(0, incr)
        except Exception as error:  # the real API replies with error_msg
            return {**self._response(), "error_msg": str(error)}
        self._play_opponent()
        return self._response()

    def _apply(self, player: int, token: str) -> None:
        engine = self.engine
        assert engine is not None
        if token == "f":
            engine.act(player, "fold")
        elif token == "k":
            engine.act(player, "check")
        elif token == "c":
            engine.act(player, "call")
        elif token.startswith("b"):
            target = int(token[1:])
            legal = engine.legal_actions(player)
            if target >= int(legal["raise_max"]):
                engine.act(player, "all_in")
            else:
                engine.act(player, "raise", max(target, int(legal["raise_min"])))
        else:
            raise ValueError(f"bad token {token!r}")

    def _play_opponent(self) -> None:
        engine = self.engine
        assert engine is not None
        guard = 0
        while not engine.hand_complete and engine.current_player == 1 and guard < 40:
            guard += 1
            legal = engine.legal_actions(1)
            roll = self.rng.random()
            if self.policy == "passive" or roll < 0.45:
                if legal.get("check"):
                    engine.act(1, "check")
                elif legal.get("call"):
                    engine.act(1, "call")
                else:  # pragma: no cover
                    engine.act(1, "fold")
            elif roll < 0.9 and legal.get("raise"):
                minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
                span = max(maximum - minimum, 0)
                engine.act(1, "raise", minimum + int(span * self.rng.random() * 0.35))
            elif legal.get("fold"):
                engine.act(1, "fold")
            elif legal.get("check"):
                engine.act(1, "check")
            else:  # pragma: no cover
                engine.act(1, "call")

    def _response(self) -> dict:
        engine = self.engine
        assert engine is not None
        client_is_button = engine.button == 0
        client_pos = int(client_is_button) if self.invert_client_pos else int(not client_is_button)
        response = {
            "token": f"fake-{self.hand_index}",
            "client_pos": client_pos,
            "hole_cards": [format_card(card) for card in engine.hole_cards[0]],
            "board": [format_card(card) for card in engine.community],
            "action": _action_string(engine),
            "winnings": None,
        }
        if engine.hand_complete:
            response["winnings"] = engine.stacks[0] - STACK
        return response


class RandomAgent:
    """Serving-contract agent that exercises folds, calls, raises and all-ins."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        legal = game.legal_actions(player)
        roll = self.rng.random()
        if roll < 0.12 and legal.get("fold"):
            return 0
        if roll < 0.28 and legal.get("raise"):
            self._target = None
            return 2
        if roll < 0.33 and legal.get("all_in"):
            return 3
        return 1

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        legal = game.legal_actions(player)
        if choice == 0 and legal.get("fold"):
            game.act(player, "fold")
        elif choice == 3 and legal.get("all_in"):
            game.act(player, "all_in")
        elif choice == 2 and legal.get("raise"):
            minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
            span = max(maximum - minimum, 0)
            game.act(player, "raise", minimum + int(span * self.rng.random() * 0.5))
        elif legal.get("check"):
            game.act(player, "check")
        else:
            game.act(player, "call")

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        return None


class SlumbotHarnessNullTests(unittest.TestCase):
    def test_always_fold_from_the_button_is_exactly_minus_half_bb(self) -> None:
        """The analytic anchor: post the small blind, fold, lose exactly 0.5 bb."""
        server = FakeSlumbot(seed=3)
        report = play_match(
            ScriptedAgent("always-fold"),
            hands=80,
            aivat=False,
            progress=False,
            post=server,
        )
        self.assertEqual(report["excluded"], 0, report["errors"])
        button = report["by_position"]["button"]
        self.assertGreater(button["hands"], 20, "fake server did not deal enough button hands")
        # Exact, not approximate: every button hand loses precisely the blind.
        self.assertAlmostEqual(button["bb_per_100"], -50.0, places=6, msg=report)
        self.assertAlmostEqual(button["stdev_bb"], 0.0, places=9, msg=report)

    def test_always_fold_out_of_position_stays_within_analytic_bounds(self) -> None:
        server = FakeSlumbot(seed=5)
        report = play_match(
            ScriptedAgent("always-fold"),
            hands=80,
            aivat=False,
            progress=False,
            post=server,
        )
        low, high = FOLD_OUT_OF_POSITION_BOUNDS
        big_blind = report["by_position"]["big_blind"]
        self.assertGreater(big_blind["hands"], 20)
        self.assertGreaterEqual(big_blind["bb_per_100"], low * 100)
        self.assertLessEqual(big_blind["bb_per_100"], high * 100)

    def test_position_is_derived_from_the_protocol_not_client_pos(self) -> None:
        """Flipping the reported client_pos convention must change nothing."""
        results = []
        for invert in (False, True):
            server = FakeSlumbot(seed=3, invert_client_pos=invert)
            results.append(
                play_match(
                    ScriptedAgent("always-fold"),
                    hands=60,
                    aivat=False,
                    progress=False,
                    post=server,
                )
            )
        self.assertEqual(
            results[0]["by_position"]["button"]["bb_per_100"],
            results[1]["by_position"]["button"]["bb_per_100"],
        )
        self.assertAlmostEqual(results[1]["by_position"]["button"]["bb_per_100"], -50.0, places=6)
        # The mapping is reported so a real session settles the convention.
        self.assertNotEqual(results[0]["client_pos_mapping"], results[1]["client_pos_mapping"])

    def test_mirroring_survives_a_full_random_agent_session(self) -> None:
        """Any mirror divergence raises and shows up as an excluded hand."""
        server = FakeSlumbot(seed=11)
        report = play_match(
            RandomAgent(seed=4),
            hands=250,
            aivat=False,
            progress=False,
            post=server,
        )
        self.assertEqual(report["excluded"], 0, report["errors"])
        self.assertEqual(report["board_desyncs"], 0, report)
        self.assertEqual(report["hands"], 250)

    def test_zero_sum_against_the_server(self) -> None:
        """Reported winnings must equal the engine's own accounting, hand by hand."""
        server = FakeSlumbot(seed=13)
        agent = RandomAgent(seed=9)
        total = 0.0
        for _ in range(120):
            response = server("new_hand", {})
            mirror = MirroredHand(response)
            while response.get("winnings") is None:
                mirror.sync(response)
                if mirror.engine.hand_complete:
                    break
                incr, _event = mirror.act_for_client(agent)
                response = server("act", {"token": response["token"], "incr": incr})
                self.assertNotIn("error_msg", response, response)
            mirror.sync(response)
            engine = server.engine
            assert engine is not None
            # The client's winnings and the server engine's stack delta are the
            # same quantity measured two ways.
            self.assertEqual(response["winnings"], engine.stacks[0] - STACK)
            # The mirrored engine must agree on the public state.
            self.assertEqual(len(mirror.engine.community), len(engine.community))
            self.assertEqual(mirror.engine.community, engine.community)
            total += float(response["winnings"])
        self.assertNotEqual(total, 0.0, "a random agent tying exactly is implausible")

    def test_agent_action_mapping_is_not_reimplemented(self) -> None:
        """The token must come from the agent's own executed engine action."""
        server = FakeSlumbot(seed=17, policy="passive")
        response = server("new_hand", {})
        mirror = MirroredHand(response)
        mirror.sync(response)

        class RaiseTo(ScriptedAgent):
            def __init__(self, target: int) -> None:
                super().__init__("always-call")
                self.target = target

            def select(self, game, player):  # noqa: ANN001
                return 2

            def execute(self, game, player, choice):  # noqa: ANN001
                game.act(player, "raise", self.target)

        target = 640
        incr, event = mirror.act_for_client(RaiseTo(target))
        # Slumbot's b<amount> is a street-cumulative bet-to total, which is
        # exactly what the engine records for a raise.
        self.assertEqual(incr, f"b{target}")
        self.assertEqual(event["amount"], target)
        self.assertEqual(mirror.engine.round_bets[0], target)

    def test_all_in_encodes_as_a_bet_to_the_full_stack(self) -> None:
        server = FakeSlumbot(seed=19, policy="passive")
        response = server("new_hand", {})
        mirror = MirroredHand(response)
        mirror.sync(response)
        expected = mirror.engine.round_bets[0] + mirror.engine.stacks[0]
        incr, event = mirror.act_for_client(ScriptedAgent("always-all-in"))
        self.assertEqual(incr, f"b{expected}")
        self.assertEqual(event["action"], "raise")

    def test_deck_is_sanitized_against_the_real_cards(self) -> None:
        """Phantom deals must never duplicate a card Slumbot actually dealt."""
        server = FakeSlumbot(seed=23)
        for _ in range(40):
            response = server("new_hand", {})
            mirror = MirroredHand(response)
            mirror.sync(response)
            engine = mirror.engine
            known = set(engine.hole_cards[0]) | set(engine.community)
            self.assertFalse(known & set(engine.hole_cards[1]), "phantom opponent hand collides")
            self.assertFalse(known & set(engine.deck), "deck still holds a known card")
            self.assertEqual(len(set(engine.deck)), len(engine.deck), "duplicate cards in the deck")

    def test_excluded_hands_are_not_counted_in_the_mean(self) -> None:
        """A hand the agent cannot complete must be dropped, not folded and banked."""

        class BrokenAgent(ScriptedAgent):
            def __init__(self) -> None:
                super().__init__("always-call")

            def select(self, game, player):  # noqa: ANN001
                raise RuntimeError("synthetic agent failure")

        server = FakeSlumbot(seed=29)
        report = play_match(BrokenAgent(), hands=20, aivat=False, progress=False, post=server)
        # Every hand is either counted or excluded, never both and never lost.
        self.assertEqual(report["hands"] + report["excluded"], 20)
        self.assertGreater(report["excluded"], 12, report)
        self.assertIn("RuntimeError: synthetic agent failure", str(report["errors"]))
        # The only hands that may still count are those that finished before the
        # agent was ever asked to act (Slumbot folding its button preflop). A
        # conceded hand must never enter the sample, so nothing negative can.
        for name in ("button", "big_blind"):
            summary = report["by_position"][name]
            if summary["hands"]:
                self.assertGreater(
                    summary["bb_per_100"],
                    0.0,
                    f"a folded-out hand was banked in {name}: {report}",
                )


class ProtocolEncodingTests(unittest.TestCase):
    def test_tokenize_handles_streets_and_trailing_separators(self) -> None:
        self.assertEqual(tokenize("cb300c/kk/kb300c/kb1200c"), ["c", "b300", "c", "k", "k", "k", "b300", "c", "k", "b1200", "c"])
        self.assertEqual(tokenize("b200c/"), ["b200", "c"])
        self.assertEqual(tokenize(""), [])

    def test_tokenize_rejects_malformed_strings(self) -> None:
        with self.assertRaises(SlumbotError):
            tokenize("b/")
        with self.assertRaises(SlumbotError):
            tokenize("x")

    def test_card_formatting_round_trips(self) -> None:
        from backend.eval.slumbot import parse_card

        for text in ("Ac", "Td", "2s", "Kh"):
            self.assertEqual(format_card(parse_card(text)), text)


if __name__ == "__main__":
    unittest.main()
