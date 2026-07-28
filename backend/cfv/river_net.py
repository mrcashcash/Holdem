"""River CFV network: model, dataset and training (P3a).

Architecture is DeepStack's (7 x 500 PReLU, exact zero-sum output projection),
reusing `model.zero_sum_project`. Three deliberate differences from CFV v0:

* **Exact per-combo I/O, not 169 buckets.** v0 used 169 per player; DeepStack and
  Supremus use ~1,000. Only ~1,081 combos are live on a river board, so per-combo
  I/O is barely wider than the literature's and removes a source of doubt. v0's
  "raw-combo cannot generalize" result was measured at 7,750 samples, where
  nothing could — it is a sample-size finding, not a representation one, and it
  is re-tested here against a zero-predictor baseline.
* **Pot-normalised targets.** CFVs are divided by the pot, so the net learns a
  scale-free quantity and one model serves every pot size.
* **Stack-aware inputs.** Pot/stack and SPR enter as explicit scalars, so a single
  net covers 50/100/200bb rather than one model per depth (the stack-normalised
  agent decision in section 8).

The acceptance metric is NOT loss. It is (a) validation MAE against the
zero-predictor baseline that v0 failed, and (b) later, action-changing error:
how often a net-priced horizon changes the decision a full solve would make.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.solver.gpu.deals import NUM_COMBOS

def zero_sum_project(cfvs: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Exact zero-sum projection (DeepStack's outer layer): r0.f0 + r1.f1 == 0.

    Subtracts half the violation from each player, spread over their range mass,
    so the net can never break the game's zero-sum identity however wrong its raw
    estimates are.
    """
    violation = (ranges * cfvs).sum(dim=(1, 2))
    mass = ranges.sum(dim=2).clamp_min(1e-9)
    return cfvs - (violation.unsqueeze(1) / (2.0 * mass)).unsqueeze(2)


# pot/stack, spr, board multi-hot, both ranges
INPUT_DIM = 2 + 52 + 2 * NUM_COMBOS
OUTPUT_DIM = 2 * NUM_COMBOS


class RiverCfvNet(nn.Module):
    def __init__(self, hidden: int = 500, layers: int = 7) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = INPUT_DIM
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.PReLU()]
            width = hidden
        blocks.append(nn.Linear(width, OUTPUT_DIM))
        self.body = nn.Sequential(*blocks)
        self.hidden = hidden
        self.layers = layers

    def forward(self, scalars: torch.Tensor, board_hot: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
        """scalars [B,2], board_hot [B,52], ranges [B,2,C] -> [B,2,C] pot-normalised CFVs."""
        encoded = torch.cat([scalars, board_hot, ranges.reshape(ranges.shape[0], -1)], dim=1)
        cfvs = self.body(encoded).reshape(-1, 2, NUM_COMBOS)
        return zero_sum_project(cfvs, ranges)


def shard_names(directory: Path) -> list[str]:
    """Every shard across every worker's manifest, interleaved.

    Parallel generation writes one manifest per worker. Interleaving rather than
    concatenating matters: workers advance through the (stack, pot) grid at the
    same time, so taking a prefix with `limit` still spans all four depths
    instead of loading one worker's slice.
    """
    manifests = sorted(directory.glob("manifest*.json"))
    if not manifests:
        raise FileNotFoundError(f"no manifest in {directory}")
    per_worker = [json.loads(path.read_text(encoding="utf-8")).get("shards", [])
                  for path in manifests]
    names: list[str] = []
    for index in range(max((len(shards) for shards in per_worker), default=0)):
        for shards in per_worker:
            if index < len(shards):
                names.append(shards[index])
    return names


def dataset_rows(directory: Path) -> int:
    return sum(
        json.loads(path.read_text(encoding="utf-8")).get("rows", 0)
        for path in directory.glob("manifest*.json")
    )


def load_shards(directory: Path, limit: int | None = None) -> dict:
    """Load generated shards into flat arrays, pot-normalising the targets."""
    boards, scalars, ranges, values, valids = [], [], [], [], []
    rows = 0
    for name in shard_names(directory):
        payload = np.load(directory / name)
        pot = payload["pot_bb"].astype(np.float32)
        stack = payload["stack_bb"].astype(np.float32)
        board = payload["board"].astype(np.int64)
        valid = np.unpackbits(payload["valid"], axis=1)[:, :NUM_COMBOS].astype(bool)
        # Targets are pot-normalised so the net learns a scale-free quantity.
        value = payload["values"].astype(np.float32) / pot[:, None, None]
        boards.append(board)
        scalars.append(np.stack([pot / stack, pot / np.maximum(stack - pot / 2.0, 1e-3)], axis=1))
        ranges.append(payload["ranges"].astype(np.float32))
        values.append(value)
        valids.append(valid)
        rows += board.shape[0]
        if limit is not None and rows >= limit:
            break
    return {
        "board": np.concatenate(boards),
        "scalars": np.concatenate(scalars),
        "ranges": np.concatenate(ranges),
        "values": np.concatenate(values),
        "valid": np.concatenate(valids),
    }


def board_multi_hot(board: np.ndarray) -> np.ndarray:
    out = np.zeros((board.shape[0], 52), dtype=np.float32)
    np.put_along_axis(out, board, 1.0, axis=1)
    return out


def masked_mae(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.unsqueeze(1).float()
    return ((predicted - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def train(
    directory: Path,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    hidden: int = 500,
    layers: int = 7,
    validation_fraction: float = 0.1,
    device: str | None = None,
    limit: int | None = None,
    progress: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Train the river CFV net.

    With ``checkpoint_dir`` set the run is observable and restartable: the best
    net so far is written after every improving epoch, per-epoch telemetry is
    appended as JSONL, and a timestamped log records progress. Without it the
    run is in-memory only (used by the scaling probe, which trains many small
    models and wants no artefacts).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_handle = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (checkpoint_dir / "train.log").open("a", encoding="utf-8")

    def emit(message: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        if progress:
            print(stamped, flush=True)
        if log_handle is not None:
            log_handle.write(stamped + "\n")
            log_handle.flush()

    data = load_shards(directory, limit=limit)
    count = data["board"].shape[0]

    # Split BY BOARD, not by row: rows from one solve share a board and would
    # otherwise leak across the split and flatter the validation number.
    board_key = np.array([hash(tuple(sorted(row))) for row in data["board"]])
    unique = np.unique(board_key)
    rng = np.random.default_rng(0)
    rng.shuffle(unique)
    holdout = set(unique[: max(1, int(len(unique) * validation_fraction))].tolist())
    is_validation = np.array([key in holdout for key in board_key])

    tensors = {
        "scalars": torch.tensor(data["scalars"]),
        "board": torch.tensor(board_multi_hot(data["board"])),
        "ranges": torch.tensor(data["ranges"]),
        "values": torch.tensor(data["values"]),
        "valid": torch.tensor(data["valid"]),
    }
    train_index = torch.tensor(np.flatnonzero(~is_validation))
    validation_index = torch.tensor(np.flatnonzero(is_validation))

    net = RiverCfvNet(hidden=hidden, layers=layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    def evaluate(index: torch.Tensor) -> tuple[float, float]:
        net.eval()
        total, baseline, seen = 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(index), 1024):
                chunk = index[start : start + 1024]
                scalars = tensors["scalars"][chunk].to(device)
                board = tensors["board"][chunk].to(device)
                ranges = tensors["ranges"][chunk].to(device)
                target = tensors["values"][chunk].to(device)
                valid = tensors["valid"][chunk].to(device)
                predicted = net(scalars, board, ranges)
                total += float(masked_mae(predicted, target, valid)) * len(chunk)
                # The bar v0 failed: a net must beat predicting zero everywhere.
                baseline += float(masked_mae(torch.zeros_like(target), target, valid)) * len(chunk)
                seen += len(chunk)
        net.train()
        return total / max(seen, 1), baseline / max(seen, 1)

    history = []
    for epoch in range(epochs):
        permutation = train_index[torch.randperm(len(train_index))]
        running = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            chunk = permutation[start : start + batch_size]
            scalars = tensors["scalars"][chunk].to(device)
            board = tensors["board"][chunk].to(device)
            ranges = tensors["ranges"][chunk].to(device)
            target = tensors["values"][chunk].to(device)
            valid = tensors["valid"][chunk].to(device)
            predicted = net(scalars, board, ranges)
            loss = nn.functional.huber_loss(
                predicted * valid.unsqueeze(1), target * valid.unsqueeze(1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss)
            batches += 1
        validation_mae, baseline_mae = evaluate(validation_index)
        ratio = validation_mae / max(baseline_mae, 1e-9)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "val_mae": validation_mae,
            "zero_baseline_mae": baseline_mae,
            "vs_baseline": ratio,
        }
        history.append(entry)
        emit(
            f"  epoch {epoch:>3}  train {entry['train_loss']:.5f}  "
            f"val MAE {validation_mae:.5f}  zero-baseline {baseline_mae:.5f}  "
            f"ratio {ratio:.3f}"
        )
        if checkpoint_dir is not None:
            with (checkpoint_dir / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            # Save on improvement only: the best net is what the gate loads, and
            # a late-overfitting epoch must not overwrite it.
            if ratio <= min(row["vs_baseline"] for row in history):
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "hidden": hidden,
                        "layers": layers,
                        "epoch": epoch,
                        "val_mae": validation_mae,
                        "zero_baseline_mae": baseline_mae,
                        "vs_baseline": ratio,
                        "train_rows": int(len(train_index)),
                        "distinct_boards": int(len(unique)),
                    },
                    checkpoint_dir / "river_net.pt",
                )
                emit(f"    checkpoint saved (best ratio {ratio:.3f})")
    if log_handle is not None:
        log_handle.close()
    return {
        "rows": count,
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "distinct_boards": int(len(unique)),
        "history": history,
        "net": net,
    }
