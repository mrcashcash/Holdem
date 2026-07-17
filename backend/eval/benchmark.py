"""Fixed-style benchmark with duplicate deals.

Plays the blueprint agent against the scripted archetypes from the legacy
trainer (the same eight styles the old reports tracked, so results are
comparable) and reports bb/100 with a confidence interval. Every deal is
played twice with seats swapped (duplicate poker), which removes most card
luck from the estimate.

CLI:  python -m backend.eval.benchmark --hands 1000 [--styles maniac,nit]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics

from backend.agents.blueprint_agent import BlueprintAgent
from backend.poker import HeadsUpHoldem
from backend.rl_env import execute_action
from backend.styles import AUDIT_STYLES, BENCHMARK_STYLES, style_action

ALL_STYLES = tuple(BENCHMARK_STYLES) + tuple(AUDIT_STYLES)


def _play_single_hand(agent: BlueprintAgent, style: str, agent_seat: int, seed: int) -> float:
    """Play one hand from a seeded deck; return the agent's result in big blinds."""
    engine = HeadsUpHoldem(rng=random.Random(seed))
    stacks_before = list(engine.stacks)
    contributions_start = list(engine.contributions)
    del contributions_start
    safety = 0
    while not engine.hand_complete and safety < 200:
        player = engine.current_player
        if player == agent_seat:
            choice = agent.select(engine, player)
            agent.execute(engine, player, choice)
        else:
            choice = style_action(engine, player, style)
            execute_action(engine, player, choice)
        safety += 1
    if not engine.hand_complete:
        raise RuntimeError("benchmark hand did not terminate")
    delta = engine.stacks[agent_seat] - stacks_before[agent_seat]
    return delta / engine.big_blind


def benchmark_against_styles(
    agent: BlueprintAgent,
    hands_per_style: int = 500,
    styles: tuple[str, ...] = ALL_STYLES,
    seed: int = 0,
) -> dict:
    """bb/100 per style with a 95% CI, using duplicate (seat-swapped) deals."""
    results: dict[str, dict] = {}
    pairs = max(1, hands_per_style // 2)
    for style_index, style in enumerate(styles):
        samples: list[float] = []
        for pair in range(pairs):
            deal_seed = seed * 1_000_003 + style_index * 10_007 + pair
            as_button_seat = 0  # hand 1 of a fresh engine gives seat 0 the button
            first = _play_single_hand(agent, style, agent_seat=as_button_seat, seed=deal_seed)
            second = _play_single_hand(agent, style, agent_seat=1 - as_button_seat, seed=deal_seed)
            samples.append((first + second) / 2.0)  # duplicate pair: card luck cancels
        mean = statistics.fmean(samples)
        deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
        margin = 1.96 * deviation / math.sqrt(len(samples))
        results[style] = {
            "bb_per_100": round(mean * 100, 2),
            "ci_low_bb_per_100": round((mean - margin) * 100, 2),
            "ci_high_bb_per_100": round((mean + margin) * 100, 2),
            "hands": pairs * 2,
        }
    overall = [entry["bb_per_100"] for entry in results.values()]
    return {
        "styles": results,
        "mean_bb_per_100": round(statistics.fmean(overall), 2),
        "worst_style": min(results, key=lambda name: results[name]["bb_per_100"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the blueprint against scripted styles")
    parser.add_argument("--hands", type=int, default=1000, help="hands per style (played as duplicate pairs)")
    parser.add_argument("--styles", type=str, default="", help="comma-separated subset of styles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", action="store_true", help="benchmark the dense GPU blueprint instead")
    parser.add_argument("--subgame-iters", type=int, default=0, help="turn/river re-solve iterations (0 = blueprint only)")
    arguments = parser.parse_args()

    if arguments.gpu:
        from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

        agent = GpuBlueprintAgent.try_load()
        if agent is None:
            raise SystemExit("no GPU blueprint found — train with `python -m backend.solver.gpu.train` first")
        agent.subgame_search = arguments.subgame_iters > 0
        agent.subgame_iterations = arguments.subgame_iters or agent.subgame_iterations
    else:
        agent = BlueprintAgent.try_load()
    if agent is None:
        raise SystemExit("no blueprint artifacts found — train with `python -m backend.solver.blueprint` first")
    styles = tuple(s for s in arguments.styles.split(",") if s) or ALL_STYLES
    report = benchmark_against_styles(agent, hands_per_style=arguments.hands, styles=styles, seed=arguments.seed)
    for style, entry in report["styles"].items():
        print(
            f"{style:18} {entry['bb_per_100']:+9.2f} bb/100  "
            f"[{entry['ci_low_bb_per_100']:+9.2f}, {entry['ci_high_bb_per_100']:+9.2f}]  n={entry['hands']}"
        )
    print(f"{'MEAN':18} {report['mean_bb_per_100']:+9.2f} bb/100   worst: {report['worst_style']}")


if __name__ == "__main__":
    main()
