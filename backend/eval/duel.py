"""Head-to-head duel gate: challenger checkpoint vs incumbent champion.

The decisive "is the new model actually better?" measurement — the two agents
play each other directly on duplicate seat-swapped deals (card luck cancels),
so the scripted-style confound is gone entirely. Used as the milestone gate
during long training runs: every N iterations, duel the current checkpoint
against the champion and promote only on a statistically clear win.

CLI:
    python -m backend.eval.duel --data-dir backend/data/gpu_blueprint_200bb \
        --stack-bb 200 --pairs 3000 [--promote]

Verdicts (95% CI on the challenger's bb/100 edge):
    PROMOTE     CI entirely above 0 - challenger clearly better
    KEEP        CI straddles 0      - no clear difference (keep champion)
    REGRESSION  CI entirely below 0 - challenger clearly WORSE (investigate!)

With --promote, a PROMOTE verdict backs up the champion, installs the
challenger as champion.npz, writes champion_meta.json, and asks the live API
server (if any) to reload the serving agent.
"""

from __future__ import annotations

import os

# Must be set before torch initializes: the duel is CPU-only by design so it
# never competes with a GPU training run, and search stays out of the gate.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HOLDEM_SUBGAME_ITERS", "0")

import argparse
import json
import math
import random
import shutil
import statistics
import time
from pathlib import Path

from backend.api_auth import api_authorization_headers
from backend.poker import HeadsUpHoldem


_STREET_NAMES = ("preflop", "flop", "turn", "river")


def _empty_diagnostics() -> dict:
    return {
        "challenger": {
            "decisions": 0,
            "exact_nodes": 0,
            "fallbacks": 0,
            "observed_raises": 0,
            "off_tree_raises": 0,
            "off_tree_nonraises": 0,
            "translation_gap_sum": 0.0,
            "translation_gap_max": 0.0,
            "decisions_by_street": {name: 0 for name in _STREET_NAMES},
            "decisions_by_position": {"button": 0, "out_of_position": 0},
            "decisions_by_pot_bb": {"lt5": 0, "5to20": 0, "20to50": 0, "ge50": 0},
            "decisions_by_spr": {"lt1": 0, "1to3": 0, "3to6": 0, "ge6": 0},
        },
        "champion": {
            "decisions": 0,
            "exact_nodes": 0,
            "fallbacks": 0,
            "observed_raises": 0,
            "off_tree_raises": 0,
            "off_tree_nonraises": 0,
            "translation_gap_sum": 0.0,
            "translation_gap_max": 0.0,
            "decisions_by_street": {name: 0 for name in _STREET_NAMES},
            "decisions_by_position": {"button": 0, "out_of_position": 0},
            "decisions_by_pot_bb": {"lt5": 0, "5to20": 0, "20to50": 0, "ge50": 0},
            "decisions_by_spr": {"lt1": 0, "1to3": 0, "3to6": 0, "ge6": 0},
        },
        "river_search": {
            "challenger": {
                "attempts": 0,
                "resolved": 0,
                "fallbacks": 0,
                "elapsed_ms_sum": 0.0,
                "elapsed_ms_max": 0.0,
                "fallback_errors": {},
                "tree_nodes_max": 0,
                "samples": [],
            },
            "champion": {
                "attempts": 0,
                "resolved": 0,
                "fallbacks": 0,
                "elapsed_ms_sum": 0.0,
                "elapsed_ms_max": 0.0,
                "fallback_errors": {},
                "tree_nodes_max": 0,
                "samples": [],
            },
        },
        "terminal_streets": {name: {"hands": 0, "challenger_bb": 0.0} for name in _STREET_NAMES},
    }


def _translation_gap(receiver, event: dict, receiver_node: int | None = None) -> float | None:
    """Distance from an observed raise to the receiver's nearest pot fraction."""
    if event.get("action") != "raise":
        return None
    # The live engine records a shove as a raise with action_index=3. Serving
    # maps that event directly to ALL_IN, so it is an exact action rather than
    # a wildly oversized off-tree pot fraction.
    if int(event.get("action_index", -1)) == 3:
        return 0.0
    try:
        pot_before = float(event["pot_before"])
        to_call_before = float(event["to_call_before"])
        current_bet_before = float(event["current_bet_before"])
        observed = max(float(event["amount"]) - current_bet_before, 0.0) / max(
            pot_before + to_call_before, 1.0
        )
        street = int(event["street"])
        fractions = tuple(float(value) for value in receiver.tree.config.fractions(street))
        if receiver_node is not None:
            legal = receiver.tree.legal[receiver_node]
            legal_fractions = tuple(
                fraction
                for index, fraction in enumerate(fractions)
                if 3 + index < len(legal) and legal[3 + index]
            )
            if legal_fractions:
                fractions = legal_fractions
        return min(abs(observed - fraction) for fraction in fractions) if fractions else observed
    except Exception:
        return None


def _record_decision(diagnostics: dict, actor_name: str, actor, engine: HeadsUpHoldem, player: int) -> None:
    entry = diagnostics[actor_name]
    entry["decisions"] += 1
    entry["decisions_by_street"][_STREET_NAMES[engine.street]] += 1
    position = "button" if player == engine.button else "out_of_position"
    entry["decisions_by_position"][position] += 1
    pot_bb = float(engine.pot) / max(float(engine.big_blind), 1.0)
    pot_band = "lt5" if pot_bb < 5 else ("5to20" if pot_bb < 20 else ("20to50" if pot_bb < 50 else "ge50"))
    entry["decisions_by_pot_bb"][pot_band] += 1
    spr = min(float(stack) for stack in engine.stacks) / max(float(engine.pot), 1.0)
    spr_band = "lt1" if spr < 1 else ("1to3" if spr < 3 else ("3to6" if spr < 6 else "ge6"))
    entry["decisions_by_spr"][spr_band] += 1
    try:
        query = actor.strategy_for_state(engine, player)
        exact = bool(query.get("exact_match"))
    except Exception:
        exact = False
    entry["exact_nodes"] += int(exact)
    entry["fallbacks"] += int(not exact)


def _record_translation(
    diagnostics: dict,
    receiver_name: str,
    receiver,
    event: dict,
    receiver_node: int | None,
) -> None:
    gap = _translation_gap(receiver, event, receiver_node)
    if gap is None:
        if (
            getattr(receiver.tree.config, "no_limp", False)
            and int(event.get("street", -1)) == 0
            and event.get("action") == "call"
            and int(event.get("action_count_before", -1)) == 2
        ):
            diagnostics[receiver_name]["off_tree_nonraises"] += 1
        return
    entry = diagnostics[receiver_name]
    entry["observed_raises"] += 1
    entry["translation_gap_sum"] += gap
    entry["translation_gap_max"] = max(entry["translation_gap_max"], gap)
    # Abstract raise targets are rounded to integer chips. Treat the resulting
    # <=1-chip normalized error as on-tree.
    denominator = max(
        float(event.get("pot_before", 0)) + float(event.get("to_call_before", 0)),
        1.0,
    )
    entry["off_tree_raises"] += int(gap > 1.0 / denominator + 1e-9)


def _play_hand(
    challenger,
    champion,
    challenger_seat: int,
    seed: int,
    stack_bb: float,
    diagnostics: dict | None = None,
) -> float:
    engine = HeadsUpHoldem(
        initial_stack=int(round(stack_bb * 20)), small_blind=10, big_blind=20, rng=random.Random(seed)
    )
    # The constructor has ALREADY posted blinds, so engine.stacks here is short
    # by each seat's blind. Measuring from it credits every hand with the
    # posted blind — a systematic +(SB+BB)/2 per seat-swapped pair (= +75
    # bb/100 at 10/20), which masqueraded as a consistent challenger edge
    # (caught by a model-vs-itself NULL test, 2026-07-23). True winnings are
    # relative to the full starting stack.
    before = float(engine.initial_stack)
    safety = 0
    while not engine.hand_complete and safety < 200:
        player = engine.current_player
        challenger_turn = player == challenger_seat
        actor = challenger if challenger_turn else champion
        actor_name = "challenger" if challenger_turn else "champion"
        receiver = champion if challenger_turn else challenger
        receiver_name = "champion" if challenger_turn else "challenger"
        if diagnostics is not None:
            _record_decision(diagnostics, actor_name, actor, engine, player)
            # Locate the receiver's matching pre-action node now. After the
            # action the engine history has advanced, so the Phase 3 legal
            # menu at the observed state would otherwise be unavailable.
            receiver_node = receiver._locate(engine, player)
        else:
            receiver_node = None
        actions_before = len(engine.public_actions)
        choice = actor.select(engine, player)
        if (
            diagnostics is not None
            and engine.street == 3
            and getattr(actor, "exact_river_search", False)
        ):
            search = diagnostics["river_search"][actor_name]
            detail = getattr(actor, "last_river_search", None) or {}
            search["attempts"] += 1
            resolved = detail.get("status") == "resolved"
            search["resolved"] += int(resolved)
            search["fallbacks"] += int(not resolved)
            elapsed = float(
                detail.get("decision_elapsed_ms", detail.get("elapsed_ms", 0.0))
                or 0.0
            )
            search["elapsed_ms_sum"] += elapsed
            search["elapsed_ms_max"] = max(search["elapsed_ms_max"], elapsed)
            search["tree_nodes_max"] = max(
                search["tree_nodes_max"],
                int(detail.get("tree_nodes", 0) or 0),
            )
            if len(search["samples"]) < 12:
                search["samples"].append(
                    {
                        "status": detail.get("status"),
                        "elapsed_ms": elapsed,
                        "tree_nodes": int(detail.get("tree_nodes", 0) or 0),
                        "iterations": int(detail.get("iterations", 0) or 0),
                        "history_resolves": len(detail.get("history_resolves", [])),
                        "error": detail.get("error"),
                    }
                )
            if not resolved:
                error = str(detail.get("error") or "unknown fallback")
                search["fallback_errors"][error] = (
                    search["fallback_errors"].get(error, 0) + 1
                )
        actor.execute(engine, player, choice)
        if diagnostics is not None and len(engine.public_actions) > actions_before:
            _record_translation(
                diagnostics,
                receiver_name,
                receiver,
                engine.public_actions[-1],
                receiver_node,
            )
        safety += 1
    if not engine.hand_complete:
        raise RuntimeError("duel hand did not terminate")
    result = (engine.stacks[challenger_seat] - before) / engine.big_blind
    if diagnostics is not None:
        terminal = diagnostics["terminal_streets"][_STREET_NAMES[engine.street]]
        terminal["hands"] += 1
        terminal["challenger_bb"] += result
    return result


def _finalize_diagnostics(diagnostics: dict) -> dict:
    for actor_name in ("challenger", "champion"):
        entry = diagnostics[actor_name]
        decisions = max(1, entry["decisions"])
        raises = max(1, entry["observed_raises"])
        entry["exact_node_rate"] = round(entry["exact_nodes"] / decisions, 6)
        entry["fallback_rate"] = round(entry["fallbacks"] / decisions, 6)
        entry["off_tree_raise_rate"] = round(entry["off_tree_raises"] / raises, 6)
        entry["mean_translation_gap_pot"] = round(entry["translation_gap_sum"] / raises, 6)
        entry["max_translation_gap_pot"] = round(entry["translation_gap_max"], 6)
    for entry in diagnostics["terminal_streets"].values():
        hands = max(1, entry["hands"])
        entry["challenger_bb_per_100"] = round(entry.pop("challenger_bb") / hands * 100.0, 2)
    for entry in diagnostics["river_search"].values():
        attempts = max(1, entry["attempts"])
        entry["resolve_rate"] = round(entry["resolved"] / attempts, 6)
        entry["fallback_rate"] = round(entry["fallbacks"] / attempts, 6)
        entry["mean_elapsed_ms"] = round(entry.pop("elapsed_ms_sum") / attempts, 1)
        entry["max_elapsed_ms"] = round(entry["elapsed_ms_max"], 1)
    return diagnostics


def head_to_head(
    challenger,
    champion,
    stack_bb: float,
    pairs: int = 3000,
    seed: int = 4242,
    collect_diagnostics: bool = False,
    common_random_numbers: bool = False,
) -> dict:
    """Challenger's edge over the champion in bb/100 with a 95% CI.

    ``common_random_numbers`` reseeds both agents' action-sampling RNG to the same
    value before every hand. The arms then draw identical variates at identical
    infosets and diverge ONLY where their policies differ, which removes the
    dominant noise term — two stochastic copies of the same policy rolling
    different dice. Each arm's marginal randomness is unchanged, so the difference
    estimator stays unbiased.

    Measured 2026-07-27: an off-vs-off null reads exactly **+0.00 bb/100 with
    σ = 0.00** on every seed with this on, versus estimates of +49.10 / −18.65 /
    +8.06 (σ ≈ 5-6) with it off. Without it, a single-seed duel of stochastic
    agents can report a spurious "significant" result — a 100bb null read
    +34.91 [+12.27, +57.55]. Default stays False so existing gate numbers remain
    comparable; new comparisons of stochastic agents should set it True.
    """
    if stack_bb <= 0:
        raise ValueError("stack_bb must be positive")
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    samples: list[float] = []
    diagnostics = _empty_diagnostics() if collect_diagnostics else None

    def couple(hand_seed: int) -> None:
        if not common_random_numbers:
            return
        for actor in (challenger, champion):
            target = getattr(actor, "agent", actor)  # unwrap diagnostic wrappers
            if hasattr(target, "_rng"):
                target._rng = random.Random(hand_seed * 31 + 5)

    for pair in range(pairs):
        deal_seed = seed * 1_000_003 + pair
        couple(deal_seed)
        first = _play_hand(challenger, champion, 0, deal_seed, stack_bb, diagnostics)
        couple(deal_seed)
        second = _play_hand(challenger, champion, 1, deal_seed, stack_bb, diagnostics)
        samples.append((first + second) / 2.0)
    mean = statistics.fmean(samples)
    margin = 1.96 * statistics.stdev(samples) / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    low, high = (mean - margin) * 100.0, (mean + margin) * 100.0
    verdict = "PROMOTE" if low > 0 else ("REGRESSION" if high < 0 else "KEEP")
    report = {
        "mean_bb_per_100": round(mean * 100.0, 2),
        "ci_low_bb_per_100": round(low, 2),
        "ci_high_bb_per_100": round(high, 2),
        "hands": pairs * 2,
        "pairs": pairs,
        "seed": seed,
        "verdict": verdict,
    }
    if diagnostics is not None:
        report["diagnostics"] = _finalize_diagnostics(diagnostics)
    return report


def promote(data_dir: Path, result: dict, challenger_iteration: int, champion_iteration: int) -> None:
    checkpoint = data_dir / "checkpoint.npz"
    champion = data_dir / "champion.npz"
    if champion.exists():
        shutil.copy2(champion, data_dir / f"champion-{champion_iteration}-backup.npz")
    temporary = data_dir / "champion.tmp.npz"
    shutil.copy2(checkpoint, temporary)
    temporary.replace(champion)
    meta = {
        "iteration": challenger_iteration,
        "styles_mean_bb_per_100": None,
        "promotion_method": (
            f"head-to-head vs {champion_iteration} "
            f"({result['mean_bb_per_100']:+.2f} bb/100, "
            f"CI[{result['ci_low_bb_per_100']:+.2f}, {result['ci_high_bb_per_100']:+.2f}], "
            f"n={result['hands']})"
        ),
    }
    (data_dir / "champion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Ask a live server (if any) to serve the new champion; failure is fine.
    try:
        import urllib.request

        request = urllib.request.Request(
            "http://127.0.0.1:8000/api/training/reload-last",
            headers=api_authorization_headers(),
            method="POST",
        )
        urllib.request.urlopen(request, timeout=30).read()
        print("server reloaded with new champion")
    except Exception as exc:
        print(f"server reload skipped ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Duel the latest checkpoint against the champion")
    parser.add_argument("--data-dir", type=str, required=True, help="artifact dir, e.g. backend/data/gpu_blueprint_200bb")
    parser.add_argument("--stack-bb", type=float, required=True)
    parser.add_argument("--pairs", type=int, default=3000, help="duplicate pairs (hands = 2x)")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--promote", action="store_true", help="install the challenger as champion on a PROMOTE verdict")
    arguments = parser.parse_args()

    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    data_dir = Path(arguments.data_dir)
    challenger = GpuBlueprintAgent.try_load(data_dir / "checkpoint.npz")
    if challenger is None:
        raise SystemExit(f"no checkpoint at {data_dir / 'checkpoint.npz'}")
    champion = GpuBlueprintAgent.try_load(data_dir / "champion.npz")
    for agent in (challenger, champion):
        if agent is not None:
            agent.subgame_search = False

    if champion is None:
        print(f"no champion yet - installing checkpoint iter={challenger.iteration} as first champion")
        if arguments.promote:
            promote(data_dir, {"mean_bb_per_100": 0, "ci_low_bb_per_100": 0, "ci_high_bb_per_100": 0, "hands": 0}, challenger.iteration, 0)
        return

    print(f"duel: challenger iter={challenger.iteration} vs champion iter={champion.iteration} @ {arguments.stack_bb:.0f}bb")
    started = time.time()
    result = head_to_head(challenger, champion, arguments.stack_bb, pairs=arguments.pairs, seed=arguments.seed)
    print(
        f"challenger edge: {result['mean_bb_per_100']:+.2f} bb/100 "
        f"[{result['ci_low_bb_per_100']:+.2f}, {result['ci_high_bb_per_100']:+.2f}] "
        f"n={result['hands']} ({time.time() - started:.0f}s)"
    )
    print(f"VERDICT: {result['verdict']} (challenger {challenger.iteration} vs champion {champion.iteration})")
    if result["verdict"] == "PROMOTE" and arguments.promote:
        promote(data_dir, result, challenger.iteration, champion.iteration)
        print(f"promoted iter={challenger.iteration} to champion")


if __name__ == "__main__":
    main()
