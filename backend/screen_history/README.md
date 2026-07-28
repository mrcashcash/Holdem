# Screen Hand History

This package reconstructs heads-up Hold'em hand histories from pixels. It can
process one saved screenshot or continuously watch a visible browser/desktop
application without reading the DOM, calling the game API, or accessing server
state.

## Package layout

```text
backend/screen_history/
  capture.py             Window, monitor, and explicit-region capture
  layouts/coinpoker.py   CoinPoker Dealer Chat and four-color-card adapter
  stream_capture.py      Windows Graphics Capture stream and MSS fallback
  decision.py            Validated rules state to serving-brain final action
  autoplay.py            Optional verified clicking of the poker client
  runtime.py             Shared watcher lifecycle and background-safe events
  gui.py                 Tkinter capture setup and live status control panel
  recognition.py         OCR, card matching, parsing, and single-image validation
  watcher.py             Live table state, transition search, tracking, and outputs
  profiles/default.json  Normalized regions for the current simulator layout
```

The executable compatibility entry points remain under `tools/`:

```text
tools/screenshot_to_hand_history.py
tools/watch_poker_screen.py
tools/screen_history_gui.py
```

## Shared dependencies

The feature intentionally reuses only two project components:

- `backend/poker.py` is the authoritative rules engine. Recognized actions are
  replayed through it to reject illegal turn order, amounts, and street changes.
- `frontend/public/assets/casino-cards/` contains the exact 52 card templates.
  The recognizer uses their colors and artwork instead of treating cards as text.

The API application and frontend do not import this package. Running either CLI
does not restart or mutate the server.

## Installation

From the repository root:

```powershell
python -m pip install -r backend/requirements.txt
```

The relevant dependencies are OpenCV (installed through RapidOCR), RapidOCR,
ONNX Runtime, resvg_py, NumPy, `windows-capture`, and MSS. On 64-bit Windows,
`windows-capture` supplies the event-driven Windows Graphics Capture backend.
MSS remains installed as the portable fallback. `resvg_py` ships platform
wheels and does not require a separately installed Cairo DLL on Windows.

Tkinter supplies the desktop GUI and is included with the standard Windows
Python installer, so the control panel adds no separate GUI framework.

## Desktop control panel

Launch the GUI from the repository root:

```powershell
python tools/screen_history_gui.py
```

The control panel provides:

- window, monitor, or explicit desktop-region capture;
- an always-available **Select with mouse** overlay for drawing the capture
  region directly on the desktop;
- profile selection and calibration from an existing screenshot;
- capture backend, stream FPS, stability delay, blind sizes, transition-search
  depth, and output settings;
- optional serving-brain decisions with a minimum recognition confidence;
- **Test Capture**, including a screenshot preview and recognized table fields;
- **Start** and **Stop**, live status, recognition warnings, and an activity log.

Use **Refresh** after launching or renaming the simulator. For window capture,
select the application title; this works with browsers and visible native,
Electron, or Tauri desktop windows. For an app whose title changes often,
region capture is usually more reliable.

To capture only part of the desktop, click **Select with mouse**. The control
panel minimizes, then a native Windows listener records the next left-button
drag anywhere on the desktop. Releasing the mouse restores the GUI,
automatically stores the absolute coordinates, switches the source to
**Region**, and briefly shows the selected boundary in red. Press `Esc` or
right-click to cancel without changing the previous region. Selection times out
after one minute. Because the drag occurs over the real desktop, pause the
simulator and begin the drag away from an action button. The coordinate field
remains available for optional fine adjustment.

Run **Test Capture** before starting the watcher. Confirm that the preview shows
only the intended simulator and review the recognized hand number, pot, stacks,
cards, board, button, and confidence. The GUI saves its last settings to
`backend/data/screen_history_gui.json`, which is ignored by Git.

In the default **Auto** mode, Windows Graphics Capture delivers frames when the
selected window or monitor changes. The stream is capped at the selected FPS;
it is not a one-second screenshot timer. If that backend is unavailable, fails,
or produces no initial frame, the watcher reports the change and continues with
the MSS fallback at the same FPS.

Incoming frames pass through a bounded two-second circular buffer. The watcher
compares them with the last OCR milestone and waits for the changed view to stay
still for the configured stability delay (300 ms by default). Only that stable
milestone is sent to OCR. Capture therefore continues while OCR runs, without
performing expensive text recognition on every animation frame. The status
shows the active backend, delivered FPS, total frames, and stable milestones
awaiting OCR. CoinPoker retains up to 16 stable milestones in capture order;
other profiles retain 32. The separate brain queue remains latest-only, so
complete hand reconstruction does not make a recommendation wait behind an
obsolete decision request. A recognition-queue overflow is recorded and shown
as an explicit history gap instead of silently replacing a pending state.

For CoinPoker, the effective stability delay is 25 ms. At a 30 FPS capture rate,
one unchanged frame interval is enough to retain a milestone before the table
resets. The watcher locates the
visible Dealer Chat and CoinPoker windows from their desktop rectangles, caches
their geometry, and recognizes only changed action rows plus the small
name/stack/pot/card-rank regions. Card faces and suits remain OpenCV-based.
Unchanged rows reuse their prior validated text. A complete full-screen OCR pass
is retained as the fallback when the two desktop windows cannot be localized.
After the OCR models are warm, this avoids the multi-second full-monitor
detector on normal live frames.

Visual-change detection compares against the last queued state—not merely the
previous screenshot—and takes the strongest score across the street header,
pot, action status, wagers, board, and hero-card areas. Small staged card
animations therefore accumulate until queued instead of being diluted by the
large static browser window.

CoinPoker ranks are read from each localized card corner instead of accepting a
broad full-screen OCR result. High-confidence corners finish in one pass;
uncertain or conflicting glyphs are compared across color, grayscale, and
thresholded variants. Verified Hero cards are then fixed for the current hand,
verified card-corner signatures reuse their cached ranks, and community cards
are append-only. As a hand progresses, only new or visually changed cards need
rank OCR. If a later frame conflicts with verified cards or does not show the
board count required by the current street, the frame is marked transient and
cannot request a recommendation. Dealer Chat history and decision safety are
validated separately. The first stack-verified state fixes the starting stacks
for that hand; later streets reconstruct the live stacks from those values and
the authoritative Dealer Chat actions. A temporarily hidden seat plaque
therefore cannot replace the verified values with the generic `20.00` fallback
or suppress a valid turn/river recommendation. Later seat-text disagreements
remain visible warnings, while the cached baseline plus Dealer Chat actions
reconstruct the decision state. Before a stack baseline has been verified, a
disagreement still blocks the decision. Card, action-history, and unexplained
pot contradictions remain hard blockers. Live capture does not trigger the
saved-screenshot full-seat OCR fallback.

CoinPoker can temporarily display a pot total that includes the unmatched live
wager while the same wager remains visible beside the acting seat. When the
difference from the action-reconstructed pot is exactly that current wager, the
watcher treats the values as equivalent instead of blocking the Hero decision.
Other pot disagreements remain visible warnings and are not accepted.

A thin red border marks the active capture boundary after **Start**. It is drawn
just outside the captured pixels, follows a selected window when it moves or
resizes, and is click-through so it cannot block simulator controls. The border
is removed on **Stop**, capture failure, or GUI exit.

Screen capture and recognition run outside Tkinter's UI thread. Closing the GUI
while it is watching requests a clean stop and finalizes any partial hand before
the window exits.

## Brain decisions

Enable **Enable brain decisions** to feed validated live Hero turns into the
same serving-agent selection used by the poker API. The default decision
confidence minimum is 85%. A state is eligible only when recognition is stable,
both Hero cards and the street's complete board are verified, Hero is due to
act, and the rules engine has a complete, unambiguous legal history.

The brain receives a copy of the reconstructed game. Its sampled action is
executed only on that copy to obtain the final legal action and raise-to amount;
the watcher never mutates the tracked hand. Recommendations are output only
unless auto-play is explicitly enabled (see below).
The **Brain recommendation** panel displays the result, model name, selected
stack-depth blueprint when applicable, and input confidence.

The command-line watcher can instead query the already-running champion server.
In server mode it posts the validated cards, board, button, inferred starting
stacks, and ordered actions to `/api/champion/query`. The returned recommendation
is converted back to the live table's chip units and executed on a copied local
rules state for validation before it is shown. The full mixed strategy, champion
iteration, source, and request latency are stored with the decision.

CoinPoker's `2/5` chip values are normalized to the champion's `10/20`
abstraction by multiplying them by four. This preserves one big blind and
postflop bet sizes. CoinPoker's `0.4 BB` small blind differs from the trained
`0.5 BB` small blind, so opening preflop spots carry an explicit warning.

Whether a recommendation is still valid is decided by the tracked rules engine —
same hand, Hero still to act, same number of actions — and not by comparing
recognized frames. Frames change constantly for reasons that leave the poker spot
untouched: a transient frame carries no cards and no acting player, and dealing
the flop moves a great many pixels while Hero still faces the very same decision.
Judging staleness from pixels discarded good answers exactly when a street
changed, which is when a decision is most needed.

Brain work has its own one-item latest-state queue. Capture and OCR continue
while a model loads or turn/river re-solving runs. If the visible table advances
before the answer returns, the stale answer is recorded but not displayed as a
current decision. Transient, low-confidence, opponent-turn, ambiguous, and
partial states produce a `brain_skipped` record instead of a guessed action.
A skip whose reason comes from the state itself — the opponent is to act, the
hand is complete, Hero's cards were not read — cannot change for that state, so
it is remembered and not reconsidered. Every other reason is temporary and the
spot stays eligible: at the start of a hand the tracker has not rebuilt the rules
engine yet, and writing the spot off for that reason discarded the whole of
Hero's first decision. Temporary reasons are logged once per spot rather than on
every retry.
Decision staleness is based on the validated hand, street, cards, acting player,
and ordered action signature. Fluctuating OCR confidence, stack text, pot text,
or a transient stability flag cannot invalidate an answer for the same poker
spot. The Live Screen likewise keeps that matching recommendation visible until
a validated action, board, street, player turn, or new hand actually changes.

## Auto-play

Auto-play presses the poker client's own buttons for a decision the brain has
already accepted. It is off by default, requires `--brain-decisions`, and even
once enabled it only performs a dry run until live clicking is turned on
explicitly. Automating a real-money client is against most poker sites' terms of
service; that is a decision for the operator, not for this code.

```powershell
python tools/watch_poker_screen.py `
  --monitor 3 --profile coinpoker --fps 30 --capture-backend mss --blinds 2 5 `
  --brain-decisions --decision-source server `
  --auto-play                     # dry run: resolves and logs the click only
```

Add `--auto-play-live` to actually press the buttons. In the GUI, tick
**Enable auto-play** and then **Live clicking**, which asks for confirmation.
**Press F12 at any time to stop auto-play** (`--auto-play-panic-key`).

A click happens only when all of the following hold:

- the decision passed every existing gate (stable frame, verified Hero cards and
  board, Hero to act, unambiguous legal history, confidence above the decision
  minimum) and was not stale by the time it returned;
- recognition confidence is at or above the separate, stricter auto-play
  minimum (`--auto-play-min-confidence`, default 90%);
- the decision carries no *blocking* warning, unless `--auto-play-allow-warnings`
  is set. Only specific warnings block: an untrained node, a fallback strategy,
  the CoinPoker `0.4 BB` opening-blind mismatch, and locally repaired actions.
  The champion server attaches an abstraction-mapping note to every response, so
  treating any warning as a blocker would silently refuse every click;
- the action is permitted by `--auto-play-actions`;
- this exact decision fingerprint has not been clicked before and the per-hand
  and per-session caps allow it. There is no cooldown by default: consecutive
  decisions are routinely a fraction of a second apart (calling preflop, then
  acting first on the flop) and any cooldown drops those actions;
- the table window is available and in the foreground. Auto-play clicks the same
  window the recognizer is reading, inherited from it directly. CoinPoker titles
  its lobby and its tables identically (`CoinPoker` — the game name is drawn in
  the client's own title bar, not the window's), so no title rule can tell them
  apart and searching for the window independently can land on the lobby.

**No coordinate is ever replayed.** After a randomized human-scale delay — during
which the attempt is abandoned if the table advances — the action strip is
captured again, OCR'd, and the click lands on the box whose text matches the
intended action. The label is matched fuzzily (small button text often reads as
`Chec<`), but only when one action wins clearly; a label that is nearly as close
to a second action is treated as unreadable. The matched box must also sit on a
filled, saturated button, which rejects CoinPoker's dark pre-action controls
(`Check/Fold`, `Call Any`) that contain the same words. If the label is missing,
ambiguous, or flat, nothing is pressed.

A raise clicks the amount field, types the value in the client's own units,
re-reads it, and presses the button only when the amount that is actually on
screen matches — either in the field or printed on the button itself. Numbers
elsewhere on the felt cannot confirm a bet.

A click is checked against the screen immediately: the client removes the action
buttons the moment Hero acts, so a button that is still sitting there means the
press did not register, and it is pressed again (`click_attempts`, 3). Between
attempts the spot is re-checked — if the table has moved on, the retry is
abandoned rather than risking a second, unintended action. Repeated presses that
the client never accepts abort the attempt instead of clicking forever.

A raise verifies its amount the same way and retries the typing once if the
field did not take it, because a field that never took focus leaves the client's
own suggested amount in place. The button is never pressed on an amount that has
not been read back.

After a click, the validated hand history has to show the action within
`confirm_seconds` (30s — measured confirmations on a live table ran from 1s to
13.5s, because the action is only visible once the client redraws and a frame
survives the recognizer's two-frame verification). If the history records a
*different* action, auto-play stops immediately: the wrong button was pressed.
If a click simply never appears, it counts as one strike; three consecutive
strikes switch auto-play off, and any confirmation resets the count. A strike
requires evidence that nothing happened — Hero still sitting on the same spot.
When the table has clearly moved on but the history has no record of Hero acting
(the Dealer Chat dropped the row; seen live keeping the opponent's call of a bet
while losing the bet itself) the outcome is `inconclusive` and does not count
against auto-play, because the click plainly worked. A single
slow read therefore no longer ends the session. Every attempt — pressed, dry
run, refused, aborted, unconfirmed — is written to `live-events.jsonl` as an
`auto_play` event.

### Sharing the mouse

Auto-play takes the physical pointer for about half a second per click. It does
not wait for the mouse to be idle first — it takes it, and re-aims through a
stray brush. What it will not do is wrestle: every step of its approach checks
that the pointer actually went where it was put (its own moves land within a
pixel), and if a hand fights it for longer than `contest_seconds` (1s) it hands
the mouse back and reports the click as `aborted`, leaving the recommendation on
screen to click manually. It also leaves the pointer where it is afterwards if
it no longer owns it, instead of dragging it back to a stale spot.

`--auto-play-no-yield` never hands the pointer back.
`--auto-play-click-method message` avoids the conflict altogether when the
client accepts posted messages, since those never touch the pointer.

Each click is preceded by a randomized think time (**Think time (min / max s)**
in the GUI, `--auto-play-delay`, default `0.8 2.4`) so acting does not look
mechanically instant. The spot is watched throughout the pause and the click is
abandoned — not sent late — if the table moves on. Set both to `0` to click as
soon as the button is verified.

While the watcher is running, auto-play's current state is drawn on the table
itself, top-left of the captured area, alongside the recommendation banner: what
it just did and why, colour-coded (green acted, blue dry run, amber stood down,
red stopped). It follows the **Decision overlay** checkbox.

### Reading the strip quickly

Text *detection* is the expensive half of OCR — three to five seconds on this
strip, which dominated the delay between deciding and clicking. Since the client
lays the action buttons out as equal columns, auto-play recognizes each column
directly (~100–200 ms) and only falls back to detection when that read looks
unreliable. A strip with no saturated pixels is rejected outright, without OCR,
because nothing is drawn there.

Buttons carrying an amount are drawn on two lines (`Call` above `0.04`).
Recognition without detection treats a crop as a single line, so the whole
button face comes back as `Ca04`, `A04`, or `S A01` and matches nothing — which
made `call` in particular fail while `fold` and `check` worked. Each column is
therefore read whole first, then across its label band alone, which yields a
clean `Call`. Single-line buttons resolve on the first read and pay nothing.

### If the click has no effect

A client can accept the pointer moving and still ignore the button press.
Synthesized input carries an "injected" marker that a low-level mouse hook can
filter on, which is a common anti-automation measure in real-money clients. The
symptom is specific: auto-play reports `clicked`, the pointer is verifiably over
the right button, the window is in the foreground — and the table does not move,
so the confirmation window expires and auto-play switches itself off.

`--auto-play-click-method message` (**Click delivery** in the GUI) posts the
click as window messages instead. Those go straight to the window's queue and so
are not visible to a low-level hook, but they only work if the client reads the
mouse from its message queue; a client that reads raw input or checks the real
cursor will ignore them too.

If neither delivery method moves the table, the client is rejecting software
input as such, and no user-space method will work — the remaining option is a
hardware HID device (a microcontroller enumerating as a USB mouse) driven over a
serial link, which the operating system cannot distinguish from a real mouse.

Click geometry lives in the profile under `action_controls`, normalized to the
table window so it survives the client being moved or resized. Those regions only
say *where to look*; verify them against a live table with:

```powershell
python tools/calibrate_action_buttons.py
python tools/calibrate_action_buttons.py --window 2   # pick another candidate
python tools/calibrate_action_buttons.py --scan       # when the strip has moved
```

Run it with Hero facing a decision. It prints every control it can read, whether
each sits on a real button, the exact click point, and normalized bounds to paste
back into the profile. Because the lobby and a table share a window title, it
lists the candidates and lets `--window` choose; this affects the tool only, not
what the watcher clicks.

## Process one screenshot

For the most complete result, capture the Hand Replay dialog with its entire
action timeline visible:

```powershell
python tools/screenshot_to_hand_history.py C:\path\to\hand.png
```

The command creates two files beside the input image:

```text
hand.hand-history.json
hand.hand-history.txt
```

The default `auto` layout recognizes both the existing simulator/replay
screenshots and CoinPoker screenshots containing the **Dealer Chat**
window beside the table. For CoinPoker, keep all street columns and both table
seats visible. The Dealer Chat supplies the ordered actions while the table
supplies player identity, current stacks, hero cards, and the board.

CoinPoker decimal values are stored internally as hundredths (`0.02` becomes
`2`) and written back as decimals in the readable history. Starting stacks are
inferred from each visible current stack plus that player's committed chips.
The promotional **Splash** amount is deliberately excluded from the pot and
action history. Hidden opponent cards, invisible actions, and future actions
are never invented, so an active table produces a validated partial history.

Useful options:

```powershell
python tools/screenshot_to_hand_history.py hand.png `
  --layout coinpoker `
  --starting-stacks 500 488 `
  --include-ocr
```

`--layout auto` is the default. Use `--layout default` to force the original
simulator/replay recognizer or `--layout coinpoker` to diagnose a Dealer Chat
capture. Explicit CoinPoker starting stacks use hundredths, so `500` means
`5.00`. `--timeline-crop` applies to the original simulator/replay layout; the
CoinPoker adapter locates its street columns automatically.

`--timeline-crop` accepts normalized values from `0` to `1`, or pixel
coordinates. The one-shot command exits with:

- `0`: validated, complete history
- `3`: output created but partial or requiring review
- `1`: recognition failed
- `2`: invalid command input

## Watch a browser or desktop application

List visible Windows application titles:

```powershell
python tools/watch_poker_screen.py --list-windows
```

Watch the current browser simulator using automatic event-driven capture at up
to 15 FPS:

```powershell
python tools/watch_poker_screen.py --window-title "Text Hold'em" --fps 15
```

The same capture path works for Electron, Tauri, or native desktop windows:

```powershell
python tools/watch_poker_screen.py --window-title "Holdem Simulator" --fps 15
```

The title may be a substring. The window can move or resize while the watcher is
running because its client rectangle is resolved before every capture. It must
remain visible and non-minimized.

Alternative capture sources:

```powershell
python tools/watch_poker_screen.py --monitor 1
python tools/watch_poker_screen.py --region 350,200,1500,800
```

CoinPoker uses a separate Dealer Chat window, so capture the entire monitor
rather than only the table window:

```powershell
python tools/watch_poker_screen.py `
  --monitor 3 `
  --profile coinpoker `
  --fps 30 `
  --capture-backend mss `
  --blinds 2 5
```

The stream receives at most 15 frames per second, but OCR runs only after a
meaningful Dealer Chat/table change has remained stable for the configured
delay. Keep Dealer Chat and the CoinPoker table visible on the same monitor.
CoinPoker does not expose a hand number in this view, so the watcher assigns a
session-local sequence number whenever the visible timeline resets to a new
blind sequence. On a multi-monitor desktop, use the monitor number containing
both windows. Force MSS when Windows Graphics Capture maps a physical monitor
to the wrong index.

Force a backend or tune the stable-state delay when diagnosing a capture:

```powershell
python tools/watch_poker_screen.py --capture-backend windows --fps 20
python tools/watch_poker_screen.py --capture-backend mss --fps 10 --stability-ms 400
```

Enable final brain recommendations from the CLI:

```powershell
python tools/watch_poker_screen.py `
  --window-title "Text Hold'em" `
  --brain-decisions `
  --min-decision-confidence 85
```

Query the running champion server for live CoinPoker recommendations:

```powershell
python tools/watch_poker_screen.py `
  --monitor 3 `
  --profile coinpoker `
  --fps 30 `
  --capture-backend mss `
  --blinds 2 5 `
  --brain-decisions `
  --decision-source server `
  --decision-server-url http://127.0.0.1:8000 `
  --min-decision-confidence 85
```

The watcher does not start, stop, or restart the server. Recommendations are
output only unless auto-play is explicitly enabled; see **Auto-play** above. A
response is discarded if the visible table advances while the server is
evaluating the prior state.

While command-line brain decisions are enabled, the watcher also serves its
latest safe recommendation at `http://127.0.0.1:8765/latest`. The feed binds to
loopback only and accepts browser access only from a localhost origin. Open
**Hand lab** in the Text Hold'em UI to see its **Live screen** panel with the
recognized cards, board, CoinPoker-scaled action, full strategy mix, recognition
confidence, champion iteration, vision time, server time, and total live
latency. The adjacent **Verified hands and steps** ledger retains the 12 most
recent session hands. It shows every stable street/action milestone in capture
order and labels each hand as in progress, verified, verified actions, partial,
or gap. **Verified actions** means the complete visible Dealer Chat sequence was
validated but CoinPoker reset the view before a terminal result was recognized.
Stack-only OCR changes never create history gaps; they affect only whether the
current state is safe for a recommendation. Later Dealer Chat frames can recover
several actions that occurred between captures; the recovered step is labeled
rather than silently replacing the record.

The UI receives updates immediately from
`http://127.0.0.1:8765/events` using server-sent events; `/latest` remains the
automatic fallback and includes the same recent-hand ledger, so reconnecting the
browser does not clear the visible session history. When the table advances,
the previous answer is visibly marked expired and retained for 12 seconds
before it clears.

Use `--decision-feed-port 0` to disable the UI feed or choose another port with
`--decision-feed-port PORT`. The current lab UI expects the default port `8765`.

`--capture-backend auto` is the default. `--interval` is retained only for
backward compatibility and is converted to FPS. Use `--once` to inspect a
single live frame without starting the continuous watcher.

## How live reconstruction works

The continuous watcher follows this pipeline:

```text
Windows Graphics Capture events (MSS fallback)
  -> bounded two-second raw-frame buffer
  -> compare focused table regions with the last milestone
  -> wait briefly for animation to settle
  -> queue only the stable milestone
  -> locate the simulator content anchor
  -> match colored card artwork
  -> OCR only numeric/status regions
  -> construct the visible table state
  -> search legal poker transitions from the previous state
  -> record a transition only when it is unique
```

The simulator can process a hero action and immediate agent response before the
next state is rendered. The watcher therefore searches up to four legal actions
between frames by default. Increase the bound only when necessary:

```powershell
python tools/watch_poker_screen.py --max-transition-actions 6
```

Transient `Agent is acting` frames are ignored. When multiple legal sequences
produce the same visible state, the hand receives an ambiguity warning rather
than a guessed action.

## Profiles and calibration

Recognition regions are stored as percentages of the detected simulator
content, not absolute desktop pixels. The built-in profile is based on the
current table layout.

Create a runtime profile for a different browser/desktop shell:

```powershell
python tools/watch_poker_screen.py `
  --calibrate C:\path\to\desktop-table.png `
  --profile desktop
```

Use it with:

```powershell
python tools/watch_poker_screen.py `
  --window-title "Holdem Simulator" `
  --profile desktop
```

Custom profiles are written to `backend/data/screen_profiles/`, which is
ignored by Git. A major table redesign may require editing the normalized
regions in that JSON file; changing only the application shell normally does
not.

## Live output

Continuous output is written to:

```text
backend/data/screen_hand_history/
  live-events.jsonl
  hand-<number>.json
  hand-<number>.txt
```

The event stream includes recognized state changes, transient frames,
ambiguities, capture errors, brain requests/decisions/skips/stale results, and
finalized output paths. Each hand JSON retains the observations, warnings, and
accepted `decisions` used during that hand. The readable hand text also lists
accepted brain decisions.

Choose a different output directory with:

```powershell
python tools/watch_poker_screen.py --output-dir C:\poker\captured-hands
```

## Recognition boundaries

- Hidden opponent cards remain unknown until the simulator reveals them.
- Invisible actions cannot be recovered from a single still image.
- Very short states that are never rendered as stable pixels can still be
  absent; transition search recovers only sequences uniquely implied by the
  next visible state.
- A minimized window may stop producing Windows capture frames. Auto mode tries
  MSS after the initial timeout, but restoring the simulator is the reliable
  fix.
- A region spanning multiple monitors automatically uses MSS because Windows
  Graphics Capture streams one monitor at a time.
- OCR uncertainty, display filters, compression, and large visual redesigns can
  require profile tuning.
- Brain output is advisory. No action is produced when the input cannot be
  reconstructed as one unique legal Hero-turn state.
- Auto-play can only press a button it can read. A theme, resolution, or layout
  change that makes the labels unreadable stops it clicking rather than making
  it click the wrong control.

## Troubleshooting

**Window not found**

Run `--list-windows` and use a distinctive substring from the displayed title.

**Window is minimized**

Restore it. Window-title discovery intentionally refuses minimized windows.

**The activity log reports an MSS fallback**

The Windows stream was unavailable, closed, or produced no initial frame. MSS
continues automatically. Reinstall `backend/requirements.txt` if
`windows-capture` is missing, or explicitly select **MSS** if the target window
does not cooperate with Windows Graphics Capture.

**Cards are not recognized**

Use a full-resolution capture, keep browser zoom near the calibrated layout,
and lower `minimum_card_score` in a custom profile only after reviewing false
matches.

For CoinPoker screenshots, use the four-color deck: black/gray spades, red
hearts, blue diamonds, and green clubs. The adapter reads the rank with OCR and
classifies the surrounding card-face color for the suit, so an unusual
single-color deck theme may require a separate layout adapter.

**Numbers are taken from the wrong area**

Create a custom profile, then adjust its normalized region coordinates.

**Transitions become untracked**

Review `live-events.jsonl`. A missing numeric field or an ambiguous legal
sequence disables engine tracking for that hand rather than inventing history.
