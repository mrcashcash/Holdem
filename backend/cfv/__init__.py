"""River counterfactual-value network (docs/PLAN_V2_STRONGEST_PLAYER.md P3a).

The v0 turn pipeline (dataset/situations/train/model + the 169-bucket
BucketCfvNet and its NetEvaluator) was deleted on 2026-07-28. It was parked,
target-noise bound, and its 169-bucket horizon was an active trap: DeepStack
and Supremus both use ~1,000 clusters, and reusing it would have reintroduced
at the horizon the abstraction the rest of the plan removes.
"""
