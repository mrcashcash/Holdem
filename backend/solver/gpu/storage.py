"""Compact decision-table storage shared by training and serving.

The original GPU solver allocated ``[all tree nodes, max buckets, actions]``
for both regrets and strategy sums. Most tree nodes are terminal, and river
nodes need far fewer buckets than flop/turn nodes, so that layout spent most
of the RTX 3060's memory on entries CFR can never read.

``CompactTableLayout`` assigns one contiguous shard to each street. A shard
contains only that street's decision nodes and exactly that street's bucket
count. The four shards are concatenated into a plain 2-D tensor so existing
in-place CUDA graph operations (discount, clone, zero, checkpoint copy) stay
fast and simple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.solver.gpu.tree import DECISION, BettingTree

STORAGE_VERSION = 2
NUM_STREETS = 4


@dataclass(frozen=True)
class StreetShard:
    street: int
    start: int
    stop: int
    decision_nodes: np.ndarray
    bucket_count: int

    @property
    def rows(self) -> int:
        return self.stop - self.start

    @property
    def decision_count(self) -> int:
        return int(self.decision_nodes.size)


class CompactTableLayout:
    """Mapping between public tree nodes and compact table rows."""

    def __init__(self, tree: BettingTree, bucket_counts: tuple[int, int, int, int]) -> None:
        if len(bucket_counts) != NUM_STREETS or any(int(count) <= 0 for count in bucket_counts):
            raise ValueError(f"expected four positive street bucket counts, got {bucket_counts}")
        self.tree = tree
        self.bucket_counts = tuple(int(count) for count in bucket_counts)
        self.node_base = np.full(len(tree), -1, dtype=np.int64)
        self.shards: tuple[StreetShard, ...]

        shards: list[StreetShard] = []
        offset = 0
        for street, bucket_count in enumerate(self.bucket_counts):
            nodes = np.flatnonzero((tree.kind == DECISION) & (tree.street == street)).astype(
                np.int64
            )
            starts = offset + np.arange(nodes.size, dtype=np.int64) * bucket_count
            self.node_base[nodes] = starts
            stop = offset + int(nodes.size) * bucket_count
            shards.append(
                StreetShard(
                    street=street,
                    start=offset,
                    stop=stop,
                    decision_nodes=nodes,
                    bucket_count=bucket_count,
                )
            )
            offset = stop
        self.shards = tuple(shards)
        self.total_rows = offset

        decision_nodes = np.flatnonzero(tree.kind == DECISION)
        if np.any(self.node_base[decision_nodes] < 0):
            raise RuntimeError("compact layout did not map every decision node")

    @property
    def max_buckets(self) -> int:
        return max(self.bucket_counts)

    def state(self) -> dict:
        return {
            "version": STORAGE_VERSION,
            "kind": "decision-node-street-sharded",
            "bucket_counts": list(self.bucket_counts),
            "decision_counts": [shard.decision_count for shard in self.shards],
            "row_counts": [shard.rows for shard in self.shards],
            "total_rows": self.total_rows,
        }

    def legal_rows(self) -> np.ndarray:
        """Legal-action mask expanded once per stored bucket row."""
        actions = self.tree.config.num_actions
        result = np.zeros((self.total_rows, actions), dtype=bool)
        for shard in self.shards:
            if not shard.decision_count:
                continue
            result[shard.start : shard.stop] = np.repeat(
                self.tree.legal[shard.decision_nodes],
                shard.bucket_count,
                axis=0,
            )
        return result

    def compact_from_dense(self, dense: np.ndarray) -> np.ndarray:
        """Migrate a legacy ``[nodes, buckets, actions]`` checkpoint."""
        if dense.ndim != 3 or dense.shape[0] != len(self.tree):
            raise ValueError(
                f"legacy table shape {dense.shape} is incompatible with {len(self.tree)} nodes"
            )
        actions = self.tree.config.num_actions
        if dense.shape[2] != actions:
            raise ValueError(f"checkpoint has {dense.shape[2]} actions; tree expects {actions}")
        compact = np.zeros((self.total_rows, actions), dtype=dense.dtype)
        for shard in self.shards:
            if dense.shape[1] < shard.bucket_count:
                raise ValueError(
                    f"legacy checkpoint has {dense.shape[1]} buckets but street "
                    f"{shard.street} requires {shard.bucket_count}"
                )
            compact[shard.start : shard.stop] = dense[
                shard.decision_nodes, : shard.bucket_count
            ].reshape(-1, actions)
        return compact

    def dense_from_compact(self, compact: np.ndarray) -> np.ndarray:
        """Materialize a dense CPU view for small subgame/evaluation consumers."""
        self.validate_compact(compact)
        dense = np.zeros(
            (len(self.tree), self.max_buckets, self.tree.config.num_actions),
            dtype=compact.dtype,
        )
        for shard in self.shards:
            if not shard.decision_count:
                continue
            dense[
                shard.decision_nodes, : shard.bucket_count
            ] = compact[shard.start : shard.stop].reshape(
                shard.decision_count,
                shard.bucket_count,
                self.tree.config.num_actions,
            )
        return dense

    def validate_compact(self, compact: np.ndarray) -> None:
        expected = (self.total_rows, self.tree.config.num_actions)
        if compact.ndim != 2 or compact.shape != expected:
            raise ValueError(f"compact table shape {compact.shape}; expected {expected}")


class CompactStrategy:
    """Read-only normalized strategy with ndarray-like node/bucket indexing."""

    def __init__(self, layout: CompactTableLayout, values: np.ndarray) -> None:
        layout.validate_compact(values)
        self.layout = layout
        self.values = np.asarray(values)

    @classmethod
    def from_sums(cls, layout: CompactTableLayout, sums: np.ndarray) -> "CompactStrategy":
        layout.validate_compact(sums)
        legal = layout.legal_rows()
        totals = sums.sum(axis=1, keepdims=True)
        legal_counts = legal.sum(axis=1, keepdims=True).clip(min=1)
        normalized = legal.astype(np.float32) / legal_counts.astype(np.float32)
        seen = totals[:, 0] > 0
        normalized[seen] = sums[seen] / np.maximum(totals[seen], 1e-30)
        normalized *= legal
        return cls(layout, normalized)

    @property
    def size(self) -> int:
        return int(self.values.size)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) not in (2, 3):
            raise TypeError("compact strategy expects [node, bucket] or [node, bucket, action]")
        node, bucket = key[0], key[1]
        base = self.layout.node_base[int(node)]
        if base < 0:
            raise IndexError(f"node {node} is not a decision node")
        bucket_array = np.asarray(bucket)
        rows = base + bucket_array
        if np.any(bucket_array < 0):
            raise IndexError("bucket ids cannot be negative")
        street = int(self.layout.tree.street[int(node)])
        if np.any(bucket_array >= self.layout.bucket_counts[street]):
            raise IndexError(f"bucket id exceeds street {street} capacity")
        if len(key) == 3:
            return self.values[rows, key[2]]
        return self.values[rows]

    def to_dense(self) -> np.ndarray:
        return self.layout.dense_from_compact(self.values)
