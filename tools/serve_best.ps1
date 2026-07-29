# Serve the strongest measured configuration. See docs/SERVING.md.
#
# Request three flop/turn sizes and let resource admission reduce the flop to
# two or one when the live SPR, node count, or VRAM estimate requires it.
# Observed opponent sizes are inserted regardless; only the resolver's own menu
# narrows. The limits below leave the display-attached 12 GiB card usable.
$env:HOLDEM_RESOLVE_STREETS     = "flop,turn,river"
$env:HOLDEM_FLOP_SIZES          = "0.33,0.75,1.4"
$env:HOLDEM_FLOP_CAP            = "2"
$env:HOLDEM_TURN_SIZES          = "0.33,0.75,1.4"
$env:HOLDEM_TURN_CAP            = "2"
$env:HOLDEM_CONTINUAL_ITERS     = "120"
$env:HOLDEM_CONTINUAL_MIN_ITERS = "60"
$env:HOLDEM_CONTINUAL_BUDGET_MS = "45000"
$env:HOLDEM_FLOP_NODE_BUDGET    = "12000"
$env:HOLDEM_RESOLVER_MAX_VRAM_MB = "9500"
$env:HOLDEM_RESOLVER_VRAM_HEADROOM_MB = "2048"
$env:HOLDEM_SHOWDOWN_WORKSPACE_MB = "384"
$env:HOLDEM_SAFETY_PRICE_GRAPH  = "1"
$env:HOLDEM_RESOLVER_PREFETCH   = "1"
$env:HOLDEM_SESSION_RUNOUT_CACHE = "1"
$env:HOLDEM_RESOLVER_WARMUP     = "1"
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
