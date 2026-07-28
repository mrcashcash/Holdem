import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import "./ChampionLab.css";
import type {
  ChampionHistoryAction,
  ChampionQueryAction,
  ChampionQueryResult,
  ChampionSpotRequest,
  ChampionSpotState,
  LiveScreenDecisionFeed,
  LiveScreenHistoryAction,
  LiveScreenStrategyAction,
} from "./types";

type LabDraft = {
  heroCards: string[];
  board: string[];
  button: 0 | 1;
  stacks: [number, number];
  actions: ChampionHistoryAction[];
};

type SavedSpot = { id: string; name: string; draft: LabDraft };

type ChampionLabProps = {
  canUseCurrent: boolean;
  initialBusy: boolean;
  initialError: string;
  initialResult: ChampionQueryResult | null;
  onClose: () => void;
  onUseCurrent: () => void;
};

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS = [
  { code: "s", symbol: "♠", tone: "black" },
  { code: "h", symbol: "♥", tone: "red" },
  { code: "d", symbol: "♦", tone: "red" },
  { code: "c", symbol: "♣", tone: "black" },
] as const;
const ALL_CARDS = SUITS.flatMap((suit) =>
  RANKS.map((rank) => ({ id: `${rank}${suit.code}`, rank, ...suit })),
);
const EMPTY_DRAFT: LabDraft = {
  heroCards: [],
  board: [],
  button: 0,
  stacks: [2_000, 2_000],
  actions: [],
};
const SAVED_KEY = "holdem.championLab.saved.v1";
const RECENT_KEY = "holdem.championLab.recent.v1";

const cloneDraft = (draft: LabDraft): LabDraft => JSON.parse(JSON.stringify(draft));
const requestOf = (draft: LabDraft): ChampionSpotRequest => ({
  hero_cards: draft.heroCards,
  board: draft.board,
  button: draft.button,
  stacks: draft.stacks,
  actions: draft.actions,
});
const actionText = (action: ChampionHistoryAction) => {
  const actor = action.player === 0 ? "Hero" : "Villain";
  if (action.action === "raise") return `${actor} raises to ${action.amount?.toLocaleString()}`;
  return `${actor} ${action.action.replace("_", " ")}s`.replace("checks", "checks").replace("calls", "calls");
};
const historyText = (actions: ChampionHistoryAction[]) =>
  actions
    .map((action) => {
      const actor = action.player === 0 ? "hero" : "villain";
      return `${actor} ${action.action}${action.amount === undefined ? "" : ` ${action.amount}`}`;
    })
    .join("\n");
const parseHistory = (value: string): ChampionHistoryAction[] =>
  value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const match = line.match(
        /^(hero|villain|p1|p2|0|1)\s+(fold|check|call|raise|all[_ -]?in)(?:\s+(?:to\s+)?(\d+))?$/i,
      );
      if (!match) throw new Error(`Line ${index + 1} is invalid.`);
      const player: 0 | 1 = ["hero", "p1", "0"].includes(match[1].toLowerCase()) ? 0 : 1;
      const action = match[2].toLowerCase().replace(/[ -]/g, "_") as ChampionHistoryAction["action"];
      const amount = match[3] ? Number(match[3]) : undefined;
      if (action === "raise" && amount === undefined) throw new Error(`Line ${index + 1} needs a raise-to amount.`);
      return amount === undefined ? { player, action } : { player, action, amount };
    });
const readSaved = (key: string): SavedSpot[] => {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(key) ?? "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};
const cardLabel = (card: string) => {
  const suffix = card.slice(-1);
  const suit = SUITS.find(
    (entry) => entry.code === suffix.toLowerCase() || entry.symbol === suffix,
  );
  return { rank: card.slice(0, -1), symbol: suit?.symbol ?? card.slice(-1), tone: suit?.tone ?? "black" };
};
const liveAmount = (value: number | null | undefined, scale: number) => {
  if (value === null || value === undefined) return "";
  if (scale === 1) return value.toLocaleString();
  const decimals = Math.max(0, Math.ceil(Math.log10(scale)));
  return (value / scale).toFixed(decimals);
};
const liveActionLabel = (
  action: Pick<LiveScreenStrategyAction, "action" | "amount">,
  scale: number,
) => {
  const name = action.action.replace("_", " ");
  const amount = liveAmount(action.amount, scale);
  if (action.action === "raise") return `Raise to ${amount}`;
  if (action.action === "call") return `Call ${amount}`;
  if (action.action === "all_in" && amount) return `All-in ${amount}`;
  return name.charAt(0).toUpperCase() + name.slice(1);
};
const liveHistoryActionLabel = (
  action: LiveScreenHistoryAction,
  players: string[],
  scale: number,
) => {
  const actor = players[action.player] || (action.player === 0 ? "Hero" : "Opponent");
  const amount = liveAmount(action.amount, scale);
  if (action.action === "raise") return `${actor} raises to ${amount}`;
  if (action.action === "call") return `${actor} calls${amount ? ` ${amount}` : ""}`;
  if (action.action === "all_in") return `${actor} goes all-in${amount ? ` ${amount}` : ""}`;
  return `${actor} ${action.action.replace("_", " ")}s`;
};

const PRESETS: Array<{ name: string; note: string; draft: LabDraft }> = [
  {
    name: "Button opener",
    note: "100 BB · first decision",
    draft: { ...EMPTY_DRAFT, heroCards: ["As", "Kh"] },
  },
  {
    name: "Facing a 3-bet",
    note: "BTN opens · BB 3-bets",
    draft: {
      ...EMPTY_DRAFT,
      heroCards: ["Qs", "Qh"],
      actions: [
        { player: 0, action: "raise", amount: 60 },
        { player: 1, action: "raise", amount: 180 },
      ],
    },
  },
  {
    name: "Flop c-bet",
    note: "Single-raised pot",
    draft: {
      ...EMPTY_DRAFT,
      heroCards: ["As", "Kh"],
      board: ["7h", "8h", "2c"],
      actions: [
        { player: 0, action: "raise", amount: 60 },
        { player: 1, action: "call" },
        { player: 1, action: "check" },
      ],
    },
  },
  {
    name: "Turn probe",
    note: "Flop checks through",
    draft: {
      ...EMPTY_DRAFT,
      heroCards: ["Jc", "Tc"],
      board: ["9c", "6d", "2s", "Qh"],
      actions: [
        { player: 0, action: "raise", amount: 60 },
        { player: 1, action: "call" },
        { player: 1, action: "check" },
        { player: 0, action: "check" },
        { player: 1, action: "check" },
      ],
    },
  },
];

function PlayingCard({ card, onRemove }: { card: string; onRemove?: () => void }) {
  const label = cardLabel(card);
  const content = (
    <>
      <strong>{label.rank}</strong>
      <span className={label.tone}>{label.symbol}</span>
    </>
  );
  return onRemove ? (
    <button className="studio-playing-card" type="button" onClick={onRemove} title={`Remove ${card}`}>
      {content}
    </button>
  ) : (
    <div className="studio-playing-card">{content}</div>
  );
}

export default function ChampionLab({
  canUseCurrent,
  initialBusy,
  initialError,
  initialResult,
  onClose,
  onUseCurrent,
}: ChampionLabProps) {
  const [draft, setDraft] = useState<LabDraft>(cloneDraft(EMPTY_DRAFT));
  const [past, setPast] = useState<LabDraft[]>([]);
  const [future, setFuture] = useState<LabDraft[]>([]);
  const [spot, setSpot] = useState<ChampionSpotState | null>(null);
  const [result, setResult] = useState<ChampionQueryResult | null>(initialResult);
  const [error, setError] = useState(initialError);
  const [busy, setBusy] = useState(false);
  const [raiseTo, setRaiseTo] = useState(60);
  const [pickerTarget, setPickerTarget] = useState<"hero" | "board">("hero");
  const [advancedText, setAdvancedText] = useState("");
  const [saveName, setSaveName] = useState("");
  const [saved, setSaved] = useState<SavedSpot[]>(() => readSaved(SAVED_KEY));
  const [recent, setRecent] = useState<SavedSpot[]>(() => readSaved(RECENT_KEY));
  const [notice, setNotice] = useState("");
  const [useInitialResult, setUseInitialResult] = useState(Boolean(initialResult || initialError || initialBusy));
  const [liveFeed, setLiveFeed] = useState<LiveScreenDecisionFeed | null>(null);
  const [liveFeedError, setLiveFeedError] = useState("");
  const requestRevision = useRef(0);

  const usedCards = useMemo(() => new Set([...draft.heroCards, ...draft.board]), [draft.heroCards, draft.board]);
  const displayedBusy = busy || (useInitialResult && initialBusy);
  const displayedError = error || (useInitialResult ? initialError : "");
  const activeResult = result ?? (useInitialResult ? initialResult : null);

  useEffect(() => {
    if (initialResult) {
      setUseInitialResult(true);
      setResult(initialResult);
      setError("");
    }
  }, [initialResult]);

  useEffect(() => {
    let active = true;
    let fallbackInterval: number | null = null;
    const refreshLiveFeed = async () => {
      try {
        const next = await api.liveScreenDecision();
        if (!active) return;
        setLiveFeed(next);
        setLiveFeedError("");
      } catch {
        if (!active) return;
        setLiveFeed(null);
        setLiveFeedError("Screen watcher offline");
      }
    };
    const startFallback = () => {
      if (fallbackInterval !== null) return;
      fallbackInterval = window.setInterval(() => void refreshLiveFeed(), 700);
    };
    void refreshLiveFeed();
    const unsubscribe = api.subscribeLiveScreenDecision(
      (next) => {
        if (!active) return;
        setLiveFeed(next);
        setLiveFeedError("");
        if (fallbackInterval !== null) {
          window.clearInterval(fallbackInterval);
          fallbackInterval = null;
        }
      },
      () => {
        if (!active) return;
        startFallback();
      },
    );
    return () => {
      active = false;
      unsubscribe();
      if (fallbackInterval !== null) {
        window.clearInterval(fallbackInterval);
      }
    };
  }, []);

  const commit = (next: LabDraft) => {
    setPast((entries) => [...entries.slice(-39), cloneDraft(draft)]);
    setFuture([]);
    setDraft(cloneDraft(next));
    setUseInitialResult(false);
    setResult(null);
    setError("");
    setNotice("");
  };

  const undo = () => {
    const previous = past[past.length - 1];
    if (!previous) return;
    setFuture((entries) => [cloneDraft(draft), ...entries]);
    setPast((entries) => entries.slice(0, -1));
    setDraft(cloneDraft(previous));
    setResult(null);
  };

  const redo = () => {
    const next = future[0];
    if (!next) return;
    setPast((entries) => [...entries, cloneDraft(draft)]);
    setFuture((entries) => entries.slice(1));
    setDraft(cloneDraft(next));
    setResult(null);
  };

  useEffect(() => {
    const revision = ++requestRevision.current;
    if (draft.heroCards.length !== 2) {
      setSpot(null);
      setBusy(false);
      return;
    }
    setBusy(true);
    setError("");
    void api
      .previewChampionSpot(requestOf(draft))
      .then(async (nextSpot) => {
        if (revision !== requestRevision.current) return;
        setSpot(nextSpot);
        const queryableBoard = [0, 3, 4, 5].includes(draft.board.length);
        const canQuery =
          nextSpot.current_player === 0 &&
          !nextSpot.complete &&
          nextSpot.required_board_count === null &&
          queryableBoard;
        if (!canQuery) {
          setResult(null);
          return;
        }
        const nextResult = await api.queryChampion(requestOf(draft));
        if (revision !== requestRevision.current) return;
        setResult(nextResult);
        const recentEntry: SavedSpot = {
          id: JSON.stringify(requestOf(draft)),
          name: `${nextSpot.street} · ${draft.heroCards.join(" ")}`,
          draft: cloneDraft(draft),
        };
        setRecent((entries) => {
          const next = [recentEntry, ...entries.filter((entry) => entry.id !== recentEntry.id)].slice(0, 6);
          window.localStorage.setItem(RECENT_KEY, JSON.stringify(next));
          return next;
        });
      })
      .catch((reason) => {
        if (revision !== requestRevision.current) return;
        setSpot(null);
        setResult(null);
        setError(reason instanceof Error ? reason.message : "Could not reconstruct this branch.");
      })
      .finally(() => {
        if (revision === requestRevision.current) setBusy(false);
      });
  }, [draft]);

  useEffect(() => {
    const minimum = spot?.legal_actions.raise_min;
    const maximum = spot?.legal_actions.raise_max;
    if (minimum === undefined || maximum === undefined) return;
    setRaiseTo((value) => Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : minimum)));
  }, [spot]);

  const selectCard = (card: string) => {
    if (usedCards.has(card)) return;
    if (pickerTarget === "hero") {
      if (draft.heroCards.length >= 2) return;
      const heroCards = [...draft.heroCards, card];
      commit({ ...draft, heroCards });
      if (heroCards.length === 2) setPickerTarget("board");
    } else if (draft.board.length < 5) {
      commit({ ...draft, board: [...draft.board, card] });
    }
  };

  const appendAction = (action: ChampionHistoryAction["action"], amount?: number) => {
    if (spot?.current_player === null || spot?.current_player === undefined) return;
    const next: ChampionHistoryAction = amount === undefined
      ? { player: spot.current_player, action }
      : { player: spot.current_player, action, amount: Math.trunc(amount) };
    commit({ ...draft, actions: [...draft.actions, next] });
  };

  const exploreChampionAction = (action: ChampionQueryAction) => {
    appendAction(action.action as ChampionHistoryAction["action"], action.action === "raise" ? action.amount ?? undefined : undefined);
  };

  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    commit(cloneDraft(preset.draft));
    setPickerTarget("board");
  };

  const saveSpot = () => {
    const name = saveName.trim() || `${spot?.street ?? "Spot"} · ${draft.heroCards.join(" ")}`;
    const entry: SavedSpot = { id: crypto.randomUUID(), name, draft: cloneDraft(draft) };
    const next = [entry, ...saved].slice(0, 20);
    setSaved(next);
    window.localStorage.setItem(SAVED_KEY, JSON.stringify(next));
    setSaveName("");
    setNotice("Spot saved locally.");
  };

  const loadSpot = (entry: SavedSpot) => commit(cloneDraft(entry.draft));
  const removeSaved = (id: string) => {
    const next = saved.filter((entry) => entry.id !== id);
    setSaved(next);
    window.localStorage.setItem(SAVED_KEY, JSON.stringify(next));
  };

  const exportSpot = async () => {
    const text = JSON.stringify(requestOf(draft), null, 2);
    setAdvancedText(text);
    await navigator.clipboard?.writeText(text);
    setNotice("Spot JSON copied.");
  };

  const importSpot = () => {
    try {
      if (advancedText.trim().startsWith("{")) {
        const parsed = JSON.parse(advancedText) as ChampionSpotRequest;
        commit({
          heroCards: parsed.hero_cards ?? [],
          board: parsed.board ?? [],
          button: parsed.button ?? 0,
          stacks: (parsed.stacks?.slice(0, 2) as [number, number]) ?? [2_000, 2_000],
          actions: parsed.actions ?? [],
        });
      } else {
        commit({ ...draft, actions: parseHistory(advancedText) });
      }
      setNotice("Imported into the workbench.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Import is invalid.");
    }
  };

  const legal = spot?.legal_actions ?? {};
  const actor = spot?.current_player === 0 ? "Hero" : spot?.current_player === 1 ? "Villain" : null;
  const raisePresets = spot && legal.raise
    ? [0.33, 0.5, 0.75, 1].map((fraction) => {
        const minimum = legal.raise_min ?? 0;
        const maximum = legal.raise_max ?? minimum;
        const target = Math.min(maximum, Math.max(minimum, Math.round((legal.current_bet ?? 0) + spot.pot * fraction)));
        return { label: `${Math.round(fraction * 100)}%`, target };
      }).filter((entry, index, entries) => entries.findIndex((candidate) => candidate.target === entry.target) === index)
    : [];
  const liveDecision = liveFeed?.decision ?? null;
  const liveScale = liveFeed?.amount_scale || 1;
  const liveStatus = liveFeedError
    ? "offline"
    : liveFeed?.status ?? "connecting";
  const liveRecommended = liveDecision?.strategy.find(
    (entry) =>
      entry.action === liveDecision.action &&
      (liveDecision.amount === null || entry.amount === liveDecision.amount),
  ) ?? liveDecision?.strategy[0];
  const liveHistory = [...(liveFeed?.history ?? [])].reverse();

  return (
    <div className="studio-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="studio-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-title" onMouseDown={(event) => event.stopPropagation()}>
        <header className="studio-header">
          <div>
            <p className="eyebrow">PROMOTED STRATEGY · BRANCH NAVIGATOR</p>
            <h2 id="studio-title">Champion Strategy Studio</h2>
          </div>
          <div className="studio-header-actions">
            <button type="button" onClick={undo} disabled={!past.length}>Undo</button>
            <button type="button" onClick={redo} disabled={!future.length}>Redo</button>
            <button type="button" onClick={() => commit(cloneDraft(EMPTY_DRAFT))}>Reset</button>
            <button className="ghost" type="button" onClick={onClose}>Close</button>
          </div>
        </header>

        <div className="studio-layout">
          <section className="studio-composer">
            <div className="studio-section-heading">
              <div>
                <span>01 · Compose</span>
                <h3>Build the spot visually</h3>
              </div>
              <button type="button" onClick={() => { setUseInitialResult(true); onUseCurrent(); }} disabled={!canUseCurrent || displayedBusy}>Use live table</button>
            </div>

            <div className="studio-presets">
              {PRESETS.map((preset) => (
                <button key={preset.name} type="button" onClick={() => applyPreset(preset)}>
                  <strong>{preset.name}</strong><small>{preset.note}</small>
                </button>
              ))}
            </div>

            <div className="studio-table">
              <div className="studio-seat studio-villain-seat">
                <span>Villain {draft.button === 1 ? "· BTN" : "· BB"}</span>
                <strong>{(spot?.stacks[1] ?? draft.stacks[1]).toLocaleString()}</strong>
                {spot && <small>bet {spot.round_bets[1].toLocaleString()}</small>}
              </div>
              <div className="studio-pot"><span>{spot?.street ?? "preflop"}</span><strong>{(spot?.pot ?? 30).toLocaleString()}</strong><small>pot</small></div>
              <div className="studio-board" onClick={() => setPickerTarget("board")}>
                {draft.board.length ? draft.board.map((card, index) => (
                  <PlayingCard key={`${card}-${index}`} card={card} onRemove={() => commit({ ...draft, board: draft.board.filter((_, position) => position !== index) })} />
                )) : <button type="button" onClick={() => setPickerTarget("board")}>Choose board</button>}
              </div>
              <div className="studio-seat studio-hero-seat">
                <span>Hero {draft.button === 0 ? "· BTN" : "· BB"}</span>
                <strong>{(spot?.stacks[0] ?? draft.stacks[0]).toLocaleString()}</strong>
                {spot && <small>bet {spot.round_bets[0].toLocaleString()}</small>}
              </div>
              <div className="studio-hole-cards" onClick={() => setPickerTarget("hero")}>
                {draft.heroCards.map((card, index) => (
                  <PlayingCard key={`${card}-${index}`} card={card} onRemove={() => commit({ ...draft, heroCards: draft.heroCards.filter((_, position) => position !== index) })} />
                ))}
                {draft.heroCards.length < 2 && <button type="button" onClick={() => setPickerTarget("hero")}>+ card</button>}
              </div>
            </div>

            <div className="studio-config">
              <label><span>Position</span><select value={draft.button} onChange={(event) => commit({ ...draft, button: Number(event.target.value) as 0 | 1 })}><option value={0}>Hero button</option><option value={1}>Villain button</option></select></label>
              <label><span>Hero stack</span><input type="number" min="20" value={draft.stacks[0]} onChange={(event) => commit({ ...draft, stacks: [Number(event.target.value), draft.stacks[1]] })} /></label>
              <label><span>Villain stack</span><input type="number" min="20" value={draft.stacks[1]} onChange={(event) => commit({ ...draft, stacks: [draft.stacks[0], Number(event.target.value)] })} /></label>
            </div>

            <div className={`studio-card-picker target-${pickerTarget}`}>
              <header><span>{pickerTarget === "hero" ? "Choose Hero cards" : "Choose board cards"}</span><small>{usedCards.size} / 52 used</small></header>
              {SUITS.map((suit) => (
                <div className="studio-deck-row" key={suit.code}>
                  <b className={suit.tone}>{suit.symbol}</b>
                  {ALL_CARDS.filter((card) => card.code === suit.code).map((card) => (
                    <button key={card.id} className={card.tone} type="button" disabled={usedCards.has(card.id) || (pickerTarget === "hero" ? draft.heroCards.length >= 2 : draft.board.length >= 5)} onClick={() => selectCard(card.id)}>{card.rank}</button>
                  ))}
                </div>
              ))}
            </div>
          </section>

          <section className="studio-branch">
            <div className="studio-section-heading"><div><span>02 · Navigate</span><h3>Walk the action tree</h3></div></div>
            {spot?.required_board_count !== null && spot?.required_board_count !== undefined ? (
              <div className="studio-board-gate"><strong>Deal the {spot.street}</strong><span>Choose {spot.required_board_count - draft.board.length} more board card(s) to continue.</span><button type="button" onClick={() => setPickerTarget("board")}>Open deck</button></div>
            ) : actor ? (
              <div className={`studio-actor ${spot?.current_player === 0 ? "hero" : "villain"}`}><span>Action on</span><strong>{actor}</strong><small>{spot?.to_call ? `${spot.to_call.toLocaleString()} to call` : "can check"}</small></div>
            ) : draft.heroCards.length < 2 ? (
              <div className="studio-board-gate"><strong>Start with two cards</strong><span>Pick Hero's hole cards from the deck.</span></div>
            ) : null}

            {spot && actor && !spot.required_board_count && !spot.complete && (
              <div className="studio-action-controls">
                <div className="studio-primary-actions">
                  {legal.fold && <button type="button" onClick={() => appendAction("fold")}>Fold</button>}
                  {legal.check && <button type="button" onClick={() => appendAction("check")}>Check</button>}
                  {legal.call && <button type="button" onClick={() => appendAction("call")}>Call <strong>{spot.to_call.toLocaleString()}</strong></button>}
                  {legal.all_in && <button type="button" onClick={() => appendAction("all_in")}>All-in</button>}
                </div>
                {legal.raise && (
                  <div className="studio-raise-control">
                    <div>{raisePresets.map((preset) => <button key={preset.label} type="button" onClick={() => setRaiseTo(preset.target)}>{preset.label}</button>)}</div>
                    <label><span>Raise to</span><input type="number" min={legal.raise_min} max={legal.raise_max} value={raiseTo} onChange={(event) => setRaiseTo(Number(event.target.value))} /></label>
                    <input type="range" min={legal.raise_min} max={legal.raise_max} value={raiseTo} onChange={(event) => setRaiseTo(Number(event.target.value))} />
                    <button className="accent" type="button" onClick={() => appendAction("raise", raiseTo)}>Raise to {raiseTo.toLocaleString()}</button>
                  </div>
                )}
              </div>
            )}

            {spot?.complete && <div className="studio-terminal"><strong>Branch complete</strong><span>{spot.result ?? "No further action."}</span><button type="button" onClick={undo}>Back up one action</button></div>}

            <div className="studio-timeline">
              <header><span>Action timeline</span><small>{draft.actions.length} actions</small></header>
              <div className="studio-blinds"><span>Hero/Villain post blinds</span><small>{draft.button === 0 ? "Hero acts first" : "Villain acts first"}</small></div>
              {draft.actions.map((action, index) => (
                <button key={`${index}-${action.action}`} type="button" onClick={() => commit({ ...draft, actions: draft.actions.slice(0, index) })} title="Rewind to before this action"><i>{index + 1}</i><span>{actionText(action)}</span><small>rewind</small></button>
              ))}
            </div>

            <details className="studio-advanced">
              <summary>Advanced import / export</summary>
              <textarea rows={7} value={advancedText} onChange={(event) => setAdvancedText(event.target.value)} placeholder={historyText(draft.actions) || "hero raise 60\nvillain call"} />
              <div><button type="button" onClick={importSpot}>Import</button><button type="button" onClick={() => void exportSpot()}>Copy spot JSON</button></div>
            </details>
          </section>

          <section className="studio-strategy" aria-live="polite">
            <div className={`studio-live-screen status-${liveStatus}`}>
              <header>
                <div>
                  <span>Live screen</span>
                  <strong>CoinPoker recommendation</strong>
                </div>
                <em>{liveStatus}</em>
              </header>
              {liveDecision ? (
                <>
                  <div className="studio-live-cards">
                    <div>
                      <span>Hero</span>
                      <div>
                        {liveDecision.hero_cards.map((card) => (
                          <PlayingCard key={`live-hero-${card}`} card={card} />
                        ))}
                      </div>
                    </div>
                    <div>
                      <span>{liveDecision.street} board</span>
                      <div>
                        {liveDecision.board.length ? (
                          liveDecision.board.map((card) => (
                            <PlayingCard key={`live-board-${card}`} card={card} />
                          ))
                        ) : (
                          <small>Preflop</small>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="studio-live-recommendation">
                    <span>
                      {liveStatus === "stale"
                        ? "Last validated action · expired"
                        : "Server action"}
                    </span>
                    <strong>
                      {liveActionLabel(
                        {
                          action: liveDecision.action,
                          amount: liveDecision.amount,
                        },
                        liveScale,
                      )}
                    </strong>
                    <small>
                      {liveRecommended
                        ? `${(liveRecommended.probability * 100).toFixed(1)}%`
                        : "validated"}
                    </small>
                  </div>
                  <div className="studio-live-meta">
                    <span>Pot {liveAmount(liveDecision.pot, liveScale)}</span>
                    <span>
                      Confidence {(liveDecision.recognition_confidence * 100).toFixed(0)}%
                    </span>
                    {liveDecision.total_latency_ms != null && (
                      <span>Live {liveDecision.total_latency_ms} ms</span>
                    )}
                    {liveFeed?.table?.recognition_ms !== null &&
                      liveFeed?.table?.recognition_ms !== undefined && (
                        <span>Vision {liveFeed.table.recognition_ms} ms</span>
                      )}
                    {liveDecision.latency_ms !== null && (
                      <span>Server {liveDecision.latency_ms} ms</span>
                    )}
                    {liveDecision.iteration !== null && (
                      <span>Iteration {liveDecision.iteration.toLocaleString()}</span>
                    )}
                  </div>
                  {liveStatus === "stale" && (
                    <p className="studio-live-stale-note">
                      The table is transitioning. Keep this only as a record—wait
                      for a green READY label before acting.
                    </p>
                  )}
                  <div className="studio-live-mix">
                    {liveDecision.strategy.map((action) => (
                      <div key={`live-${action.action}-${action.amount ?? "none"}`}>
                        <span>{liveActionLabel(action, liveScale)}</span>
                        <strong>{(action.probability * 100).toFixed(1)}%</strong>
                        <i>
                          <b style={{ width: `${action.probability * 100}%` }} />
                        </i>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="studio-live-empty">
                  <strong>
                    {liveStatus === "offline"
                      ? "Watcher not connected"
                      : liveStatus === "thinking"
                        ? "Champion is thinking…"
                        : liveStatus === "stale"
                          ? "Previous answer expired"
                          : "Waiting for Hero"}
                  </strong>
                  <span>
                    {liveFeedError ||
                      liveFeed?.message ||
                      "Start the screen watcher with server decisions enabled."}
                  </span>
                </div>
              )}
            </div>

            <section className="studio-live-history" aria-label="Verified live hand history">
              <header>
                <div>
                  <span>Live history</span>
                  <strong>Verified hands and steps</strong>
                </div>
                <small>
                  {liveHistory.length
                    ? `${liveHistory.length} recent hand${liveHistory.length === 1 ? "" : "s"}`
                    : "Waiting"}
                </small>
              </header>
              {liveHistory.length ? (
                <div className="studio-live-history-list">
                  {liveHistory.map((hand, handIndex) => (
                    <details key={hand.id} open={handIndex === 0}>
                      <summary>
                        <div>
                          <strong>Hand #{hand.hand_number ?? "?"}</strong>
                          <span>{hand.hero_cards.length ? hand.hero_cards.join(" ") : "Cards pending"}</span>
                        </div>
                        <em className={`history-status status-${hand.status}`}>
                          {hand.status.replace("_", " ")}
                        </em>
                      </summary>
                      <p className="studio-history-verification">{hand.verification_message}</p>
                      <ol>
                        {hand.steps.map((step, stepIndex) => {
                          const previousActionCount =
                            stepIndex > 0 ? hand.steps[stepIndex - 1].actions.length : 0;
                          const newActions = step.recovered
                            ? step.actions
                            : step.actions.slice(previousActionCount);
                          return (
                            <li
                              key={step.id}
                              className={step.verified ? "verified" : "unverified"}
                            >
                              <div className="studio-history-step-heading">
                                <span>{step.street ?? "unknown"}</span>
                                <strong>Pot {liveAmount(step.pot, liveScale) || "?"}</strong>
                                <small>
                                  {step.recovered
                                    ? "recovered"
                                    : step.verified
                                      ? "verified"
                                      : "check required"}
                                </small>
                              </div>
                              <div className="studio-history-board">
                                {step.board.length
                                  ? step.board.map((card) => <i key={`${step.id}-${card}`}>{card}</i>)
                                  : <i>preflop</i>}
                              </div>
                              {newActions.length > 0 && (
                                <ul>
                                  {newActions.map((action, actionIndex) => (
                                    <li key={`${step.id}-action-${actionIndex}`}>
                                      {liveHistoryActionLabel(action, hand.players, liveScale)}
                                    </li>
                                  ))}
                                </ul>
                              )}
                              {step.decision && (
                                <div className="studio-history-decision">
                                  <span>Hero decision</span>
                                  <strong>
                                    {liveActionLabel(
                                      {
                                        action: step.decision.action,
                                        amount: step.decision.amount,
                                      },
                                      liveScale,
                                    )}
                                  </strong>
                                </div>
                              )}
                              {!step.verified && step.warnings.length > 0 && (
                                <p>{step.warnings[0]}</p>
                              )}
                            </li>
                          );
                        })}
                      </ol>
                    </details>
                  ))}
                </div>
              ) : (
                <div className="studio-live-history-empty">
                  Stable CoinPoker hands will appear here in capture order.
                </div>
              )}
              {(liveFeed?.history_gap_count ?? 0) > 0 && (
                <p className="studio-history-gap-count">
                  Capture gaps detected this session: {liveFeed?.history_gap_count}
                </p>
              )}
            </section>

            <div className="studio-section-heading"><div><span>03 · Inspect</span><h3>Champion strategy</h3></div>{activeResult && <em>iteration {activeResult.iteration.toLocaleString()}</em>}</div>
            {displayedBusy ? <div className="studio-empty">Reconstructing branch…</div> : displayedError ? <div className="studio-error" role="alert">{displayedError}</div> : activeResult ? (
              <>
                <div className="studio-recommendation"><span>Most frequent</span><strong>{activeResult.recommended.label}</strong><small>{activeResult.recommended.percentage.toFixed(1)}%</small></div>
                <div className="studio-strategy-mix">
                  {activeResult.actions.map((action) => (
                    <button key={`${action.action}-${action.amount ?? "none"}`} type="button" onClick={() => exploreChampionAction(action)} disabled={!spot} title={spot ? "Play this action and continue the branch" : "Live-table results cannot be branched without their complete action history"}>
                      <div><span>{action.label}</span><strong>{action.percentage.toFixed(1)}%</strong></div>
                      <div className="studio-mix-track"><i style={{ width: `${action.percentage}%` }} /></div>
                      <small>Explore branch →</small>
                    </button>
                  ))}
                </div>
                {spot && <div className="studio-metrics"><div><span>SPR</span><strong>{spot.metrics.spr.toFixed(2)}</strong></div><div><span>Pot odds</span><strong>{spot.metrics.pot_odds_percent.toFixed(1)}%</strong></div><div><span>Effective</span><strong>{spot.metrics.effective_stack_bb.toFixed(1)} BB</strong></div><div><span>Made hand</span><strong>{spot.metrics.hand_strength ?? "Preflop"}</strong></div><div><span>Node</span><strong>{activeResult.node ?? "—"}</strong></div><div><span>Bucket</span><strong>{activeResult.bucket ?? "—"}</strong></div></div>}
                <div className="studio-warning">{activeResult.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>
              </>
            ) : spot?.current_player === 1 ? <div className="studio-empty"><strong>Villain's turn</strong><span>Choose a legal Villain action to reach the next Hero decision.</span></div> : <div className="studio-empty"><strong>No strategy yet</strong><span>Choose cards and walk the branch until Hero is due to act.</span></div>}

            <div className="studio-library">
              <header><span>Spot library</span><small>stored in this browser</small></header>
              <div className="studio-save"><input value={saveName} onChange={(event) => setSaveName(event.target.value)} placeholder="Name this spot" /><button type="button" onClick={saveSpot} disabled={draft.heroCards.length !== 2}>Save</button></div>
              {notice && <p>{notice}</p>}
              {!!saved.length && <div className="studio-saved-list">{saved.map((entry) => <div key={entry.id}><button type="button" onClick={() => loadSpot(entry)}><strong>{entry.name}</strong><small>{entry.draft.actions.length} actions</small></button><button type="button" onClick={() => removeSaved(entry.id)} aria-label={`Delete ${entry.name}`}>×</button></div>)}</div>}
              {!!recent.length && <details><summary>Recent queries ({recent.length})</summary>{recent.map((entry) => <button key={entry.id} type="button" onClick={() => loadSpot(entry)}>{entry.name}</button>)}</details>}
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}
