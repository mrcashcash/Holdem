import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { api } from "./api";
import ChampionLab from "./ChampionLab";
import GameSettingsDialog from "./GameSettingsDialog";
import {
  loadSoundSettings,
  type SoundSettings,
  sound,
  storeSoundSettings,
} from "./sound";
import type {
  CashReloadRequest,
  ChampionQueryResult,
  GameSettings,
  GameState,
  LegalActions,
  PlayerSessionStats,
  TrainingStatus,
} from "./types";

const MIN_EPISODES = 10;
const MAX_EPISODES = 500_000;
const AGENT_THINK_MIN_MS = 900;
const AGENT_THINK_MAX_MS = 2_200;

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(Math.max(value, minimum), maximum);
const cardAssetPath = (card: string) => {
  const suit = card.slice(-1);
  const suitCode = { "♥": "h", "♦": "d", "♣": "c", "♠": "s" }[suit];
  return suitCode ? `/assets/casino-cards/${card.slice(0, -1)}${suitCode}.svg` : "";
};
type ChipColour = "white" | "green" | "red" | "black" | "purple";
type ChipGroup = { value: number; colour: ChipColour; count: number };
type ChipFlight = {
  id: number;
  target: 0 | 1;
  motion: "to-pot" | "to-player";
  value: number;
  colour: ChipColour;
  delay: number;
  offsetX: number;
  offsetY: number;
  turn: number;
};
type DealerFlight = { id: number; from: 0 | 1; to: 0 | 1 };
type TableActionPopup = {
  id: number;
  text: string;
  tone: "hero" | "opponent" | "center";
};
type HandHistory = {
  handNumber: number;
  entries: string[];
  heroCards: string[];
  opponentCards: string[];
  community: string[];
  result: string | null;
  winner: number | null;
};

type HudRate = {
  successes: number;
  opportunities: number;
};

type DerivedPlayerHudStats = {
  threeBet: HudRate;
  foldToThreeBet: HudRate;
  flopCBet: HudRate;
  foldToFlopCBet: HudRate;
  wentToShowdown: HudRate;
  wonAtShowdown: HudRate;
  wonWhenSawFlop: HudRate;
};

type ParsedHandAction = {
  player: 0 | 1;
  street: 0 | 1 | 2 | 3;
  action: "fold" | "check" | "call" | "raise";
};

const snapshotOf = (game: GameState): HandHistory => ({
  handNumber: game.hand_number,
  entries: [...game.history],
  heroCards: [...game.hero_cards],
  opponentCards: [...game.opponent_cards],
  community: [...game.community],
  result: game.result,
  winner: game.winner,
});

const STREET_BOARD_COUNT: Record<string, number> = {
  flop: 3,
  turn: 4,
  river: 5,
};
const boardCountAtStep = (entries: string[]) => {
  let count = 0;
  for (const entry of entries) {
    const match = entry.match(/^(Flop|Turn|River):/i);
    if (match) count = STREET_BOARD_COUNT[match[1].toLowerCase()] ?? count;
  }
  return count;
};
const HAND_HISTORY_STORAGE_KEY = "holdem.handHistory.v1";
const MAX_STORED_HANDS = 200;
const loadStoredHands = (): HandHistory[] => {
  try {
    const raw = window.localStorage.getItem(HAND_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HandHistory[]) : [];
  } catch {
    return [];
  }
};
const storeHands = (hands: HandHistory[]) => {
  try {
    window.localStorage.setItem(
      HAND_HISTORY_STORAGE_KEY,
      JSON.stringify(hands.slice(-MAX_STORED_HANDS)),
    );
  } catch {
    /* storage unavailable — replay stays in-memory only */
  }
};
const handOutcome = (hand: HandHistory) => {
  if (hand.winner === null) return { label: "Split pot", tone: "center" as const };
  return hand.winner === 0
    ? { label: "You won", tone: "hero" as const }
    : { label: "Agent won", tone: "opponent" as const };
};

const emptyHudRate = (): HudRate => ({ successes: 0, opportunities: 0 });
const emptyDerivedHudStats = (): DerivedPlayerHudStats => ({
  threeBet: emptyHudRate(),
  foldToThreeBet: emptyHudRate(),
  flopCBet: emptyHudRate(),
  foldToFlopCBet: emptyHudRate(),
  wentToShowdown: emptyHudRate(),
  wonAtShowdown: emptyHudRate(),
  wonWhenSawFlop: emptyHudRate(),
});

const parsedActions = (hand: HandHistory): ParsedHandAction[] => {
  let street: ParsedHandAction["street"] = 0;
  return hand.entries.flatMap((entry) => {
    if (/^Flop:/i.test(entry)) street = 1;
    else if (/^Turn:/i.test(entry)) street = 2;
    else if (/^River:/i.test(entry)) street = 3;

    const match = entry.match(
      /^Player ([12]) (folds|checks|calls\b|raises to\b)/i,
    );
    if (!match) return [];
    const verb = match[2].toLowerCase();
    const action: ParsedHandAction["action"] = verb.startsWith("fold")
      ? "fold"
      : verb.startsWith("check")
        ? "check"
        : verb.startsWith("call")
          ? "call"
          : "raise";
    return [
      {
        player: (Number(match[1]) - 1) as 0 | 1,
        street,
        action,
      },
    ];
  });
};

const derivePlayerHudStats = (
  hands: HandHistory[],
): [DerivedPlayerHudStats, DerivedPlayerHudStats] => {
  const players: [DerivedPlayerHudStats, DerivedPlayerHudStats] = [
    emptyDerivedHudStats(),
    emptyDerivedHudStats(),
  ];

  for (const hand of hands) {
    const actions = parsedActions(hand);
    const preflop = actions.filter((action) => action.street === 0);
    const flop = actions.filter((action) => action.street === 1);
    const firstRaiseIndex = preflop.findIndex(
      (action) => action.action === "raise",
    );

    if (firstRaiseIndex >= 0) {
      const opener = preflop[firstRaiseIndex].player;
      const responderIndex = preflop.findIndex(
        (action, index) => index > firstRaiseIndex && action.player !== opener,
      );
      if (responderIndex >= 0) {
        const response = preflop[responderIndex];
        players[response.player].threeBet.opportunities += 1;
        if (response.action === "raise") {
          players[response.player].threeBet.successes += 1;
          const openerResponse = preflop.find(
            (action, index) =>
              index > responderIndex && action.player === opener,
          );
          if (openerResponse) {
            players[opener].foldToThreeBet.opportunities += 1;
            if (openerResponse.action === "fold")
              players[opener].foldToThreeBet.successes += 1;
          }
        }
      }
    }

    const preflopAggressor = [...preflop]
      .reverse()
      .find((action) => action.action === "raise")?.player;
    if (preflopAggressor !== undefined && flop.length > 0) {
      let aggressorActionIndex = -1;
      let facedDonkBet = false;
      for (let index = 0; index < flop.length; index += 1) {
        const action = flop[index];
        if (action.player === preflopAggressor) {
          aggressorActionIndex = index;
          break;
        }
        if (action.action === "raise") {
          facedDonkBet = true;
          break;
        }
      }
      if (!facedDonkBet && aggressorActionIndex >= 0) {
        const aggressorAction = flop[aggressorActionIndex];
        players[preflopAggressor].flopCBet.opportunities += 1;
        if (aggressorAction.action === "raise") {
          players[preflopAggressor].flopCBet.successes += 1;
          const defender = (1 - preflopAggressor) as 0 | 1;
          const response = flop.find(
            (action, index) =>
              index > aggressorActionIndex && action.player === defender,
          );
          if (response) {
            players[defender].foldToFlopCBet.opportunities += 1;
            if (response.action === "fold")
              players[defender].foldToFlopCBet.successes += 1;
          }
        }
      }
    }

    const sawFlop = hand.entries.some((entry) => /^Flop:/i.test(entry));
    const showdown = Boolean(hand.result && !/after a fold/i.test(hand.result));
    for (const player of [0, 1] as const) {
      if (sawFlop) {
        players[player].wentToShowdown.opportunities += 1;
        players[player].wonWhenSawFlop.opportunities += 1;
        if (showdown) players[player].wentToShowdown.successes += 1;
        if (hand.winner === player)
          players[player].wonWhenSawFlop.successes += 1;
        else if (hand.winner === null)
          players[player].wonWhenSawFlop.successes += 0.5;
      }
      if (showdown) {
        players[player].wonAtShowdown.opportunities += 1;
        if (hand.winner === player)
          players[player].wonAtShowdown.successes += 1;
        else if (hand.winner === null)
          players[player].wonAtShowdown.successes += 0.5;
      }
    }
  }

  return players;
};

const hudRateText = (rate: HudRate) =>
  rate.opportunities > 0
    ? `${Math.round((rate.successes / rate.opportunities) * 100)}%`
    : "—";

const servingAgentLabel = (agent?: string) => {
  switch (agent) {
    case "MultiStackBlueprintAgent":
      return "Multi-stack GPU blueprint";
    case "GpuBlueprintAgent":
      return "GPU blueprint";
    case "BlueprintAgent":
      return "CPU blueprint";
    case "HeuristicAgent":
      return "Heuristic fallback";
    default:
      return agent ?? "—";
  }
};

const CHIP_DENOMINATIONS: { value: number; colour: ChipColour }[] = [
  { value: 500, colour: "black" },
  { value: 100, colour: "red" },
  { value: 25, colour: "green" },
  { value: 5, colour: "purple" },
  { value: 1, colour: "white" },
];
const chipAssetPath = (colour: ChipColour) =>
  `/assets/chips/chip-${colour}.png`;
const chipBreakdown = (amount: number): ChipGroup[] => {
  let remaining = Math.max(0, Math.trunc(amount));
  return CHIP_DENOMINATIONS.flatMap(({ value, colour }) => {
    const count = Math.floor(remaining / value);
    remaining %= value;
    return count > 0 ? [{ value, colour, count }] : [];
  });
};
const individualChips = (amount: number) =>
  chipBreakdown(amount).flatMap(({ count, ...chip }) =>
    Array.from({ length: count }, () => chip),
  );
const actionPopupTone = (entry: string): TableActionPopup["tone"] =>
  entry.startsWith("Player 1")
    ? "hero"
    : entry.startsWith("Player 2")
      ? "opponent"
      : "center";
const actionPopupText = (entry: string) =>
  entry
    .replace(/^Player 1\b/, "YOU")
    .replace(/^Player 2\b/, "AGENT")
    .replace(/^Hand \d+:\s*/, "NEW HAND · ");

const latestActionSound = (entries: string[]) => {
  const entry = entries.at(-1) ?? "";
  if (/\bfolds\b/i.test(entry)) return "fold" as const;
  if (/\bchecks\b/i.test(entry)) return "check" as const;
  if (/\b(calls|raises to)\b/i.test(entry)) return "chips" as const;
  return null;
};

function PlayingCard({
  card,
  hidden = false,
}: {
  card?: string;
  hidden?: boolean;
}) {
  if (hidden || !card)
    return (
      <span className="playing-card card-back" aria-label="Face-down card">
        <span>♠</span>
      </span>
    );

  const suit = card.slice(-1);
  const rawRank = card.slice(0, -1);
  const rank = rawRank === "T" ? "10" : rawRank;
  return (
    <span className="playing-card card-face" aria-label={`${rank} of ${suit}`}>
      <img src={cardAssetPath(card)} alt="" />
    </span>
  );
}

function Cards({
  cards,
  hidden = false,
  className = "",
}: {
  cards: string[];
  hidden?: boolean;
  className?: string;
}) {
  if (hidden) {
    return (
      <div className={`cards ${className}`}>
        <PlayingCard hidden />
        <PlayingCard hidden />
      </div>
    );
  }
  return (
    <div className={`cards ${className}`}>
      {cards.map((card, index) => (
        <PlayingCard card={card} key={`${card}-${index}`} />
      ))}
    </div>
  );
}

function PlayerHud({
  stats,
  hands,
  derived,
  historyHands,
  player,
}: {
  stats: PlayerSessionStats;
  hands: number;
  derived: DerivedPlayerHudStats;
  historyHands: number;
  player: string;
}) {
  const hasSample = hands > 0;
  const aggression =
    hasSample && stats.aggression !== null ? `${stats.aggression}%` : "—";

  const advancedMetrics = [
    ["3B", derived.threeBet, "3-bet"],
    ["F3B", derived.foldToThreeBet, "Fold to 3-bet"],
    ["CB", derived.flopCBet, "Flop continuation bet"],
    ["FCB", derived.foldToFlopCBet, "Fold to flop continuation bet"],
    ["WTSD", derived.wentToShowdown, "Went to showdown after seeing a flop"],
    ["W$SD", derived.wonAtShowdown, "Won money at showdown"],
    ["WWSF", derived.wonWhenSawFlop, "Won when seeing a flop"],
  ] as const;

  return (
    <dl
      className="player-hud"
      aria-label={`${player} session statistics over ${hands} completed ${hands === 1 ? "hand" : "hands"}`}
      title="Session-only statistics from completed hands"
    >
      <div>
        <dt>VPIP</dt>
        <dd>{hasSample ? `${stats.vpip}%` : "—"}</dd>
      </div>
      <div>
        <dt>PFR</dt>
        <dd>{hasSample ? `${stats.pfr}%` : "—"}</dd>
      </div>
      <div>
        <dt>AGG</dt>
        <dd>{aggression}</dd>
      </div>
      {advancedMetrics.map(([label, rate, description]) => (
        <div
          key={label}
          title={`${description}: ${rate.successes}/${rate.opportunities} opportunities`}
        >
          <dt>{label}</dt>
          <dd>{hudRateText(rate)}</dd>
        </div>
      ))}
      <div
        className="player-hud-sample"
        title={
          historyHands < hands
            ? `${historyHands} of ${hands} completed hand histories are available in this browser for advanced stats`
            : `${hands} completed hands`
        }
      >
        <dt>Hands</dt>
        <dd>
          {hands}
          {historyHands < hands && <span>*</span>}
        </dd>
      </div>
    </dl>
  );
}

function ChipStack({
  amount,
  className = "",
}: {
  amount: number;
  className?: string;
}) {
  const chips = chipBreakdown(amount);
  return (
    <div
      className={`chip-stack ${className}`}
      aria-label={`${amount.toLocaleString()} chips`}
    >
      <div className="chips value-chip-rack" aria-hidden="true">
        {chips.map((chip) => (
          <span
            className={`value-chip-stack chip-${chip.colour}`}
            key={chip.value}
            style={{ height: `${30 + (chip.count - 1) * 4}px` }}
          >
            {Array.from({ length: chip.count }, (_, index) => (
              <span
                className="value-chip"
                key={index}
                style={{ bottom: `${index * 4}px`, zIndex: index + 1 }}
              >
                <img
                  className="chip-art"
                  src={chipAssetPath(chip.colour)}
                  alt=""
                />
              </span>
            ))}
          </span>
        ))}
      </div>
      <span>{amount.toLocaleString()}</span>
    </div>
  );
}

function boundedRaise(value: number, legal: LegalActions, fallback: number) {
  const minimum = legal.raise_min;
  const maximum = legal.raise_max;
  if (minimum === undefined || maximum === undefined) return fallback;
  return clamp(
    Number.isFinite(value) ? Math.trunc(value) : minimum,
    minimum,
    maximum,
  );
}

function bigBlindFromHistory(history: string[]) {
  const match = history
    .find((entry) => /big blind/i.test(entry))
    ?.match(/big blind\s+(\d+)/i);
  return match ? Number(match[1]) : 20;
}

function App() {
  const [game, setGame] = useState<GameState | null>(null);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [raiseTo, setRaiseTo] = useState(40);
  const [episodes, setEpisodes] = useState(50_000);
  const [message, setMessage] = useState("Connecting to the table…");
  const [busy, setBusy] = useState(false);
  const [agentThinking, setAgentThinking] = useState(false);
  const [autoDealing, setAutoDealing] = useState(false);
  const [handSettling, setHandSettling] = useState(false);
  const [chipFlights, setChipFlights] = useState<ChipFlight[]>([]);
  const [hands, setHands] = useState<HandHistory[]>(loadStoredHands);
  const [replayOpen, setReplayOpen] = useState(false);
  const [replayHandNumber, setReplayHandNumber] = useState<number | null>(null);
  const [replayStep, setReplayStep] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [championOpen, setChampionOpen] = useState(false);
  const [championResult, setChampionResult] =
    useState<ChampionQueryResult | null>(null);
  const [championError, setChampionError] = useState("");
  const [championBusy, setChampionBusy] = useState(false);
  const [displayedDealer, setDisplayedDealer] = useState<0 | 1 | null>(null);
  const [dealerFlight, setDealerFlight] = useState<DealerFlight | null>(null);
  const [soundSettings, setSoundSettings] =
    useState<SoundSettings>(loadSoundSettings);
  const gameRef = useRef<GameState | null>(null);
  const chipFlightId = useRef(0);
  const dealerFlightId = useRef(0);
  const autoDealTimer = useRef<number | null>(null);
  const agentDelayTimer = useRef<number | null>(null);
  const agentTurnRunning = useRef(false);
  const betControlsRef = useRef<HTMLDivElement | null>(null);

  const cancelAutoDeal = useCallback(() => {
    if (autoDealTimer.current !== null) {
      window.clearTimeout(autoDealTimer.current);
      autoDealTimer.current = null;
    }
    setAutoDealing(false);
  }, []);

  useEffect(() => {
    sound.configure(soundSettings);
    storeSoundSettings(soundSettings);
  }, [soundSettings]);

  const acceptGame = useCallback((next: GameState, newMatch = false) => {
    const previous = gameRef.current;
    const reducedMotion = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const startsFreshHand =
      !previous || newMatch || previous.hand_number !== next.hand_number;
    const handJustCompleted = Boolean(
      previous && !previous.complete && next.complete,
    );
    const newHistoryEntries = previous
      ? next.history.slice(previous.history.length)
      : [];
    if (previous) {
      if (handJustCompleted) sound.play("win");
      else if (startsFreshHand || next.community.length > previous.community.length)
        sound.play("deal");
      else {
        const actionSound = latestActionSound(newHistoryEntries);
        if (actionSound) sound.play(actionSound);
      }
    }
    if (previous && !reducedMotion) {
      const previousBets = startsFreshHand ? [0, 0] : previous.round_bets;
      const wagerFlights = ([0, 1] as const).flatMap((target) => {
        const added = Math.max(
          0,
          next.round_bets[target] - previousBets[target],
        );
        return individualChips(added).map((chip, index) => ({
          id: ++chipFlightId.current,
          target,
          motion: "to-pot" as const,
          value: chip.value,
          colour: chip.colour,
          delay: target * 90 + index * 80,
          offsetX: ((index % 5) - 2) * 13,
          offsetY: (index % 3) * 7,
          turn: ((index % 5) - 2) * 48,
        }));
      });
      const payoutDelay =
        wagerFlights.length > 0
          ? Math.max(...wagerFlights.map((flight) => flight.delay)) + 820
          : 0;
      const payoutRecipients: (0 | 1)[] =
        next.winner === null ? [0, 1] : [next.winner as 0 | 1];
      const payoutFlights =
        !previous.complete && next.complete && next.last_pot > 0
          ? payoutRecipients.flatMap((target, recipientIndex) => {
              const share =
                Math.floor(next.last_pot / payoutRecipients.length) +
                (recipientIndex === 0
                  ? next.last_pot % payoutRecipients.length
                  : 0);
              return individualChips(share).map((chip, index) => ({
                id: ++chipFlightId.current,
                target,
                motion: "to-player" as const,
                value: chip.value,
                colour: chip.colour,
                delay: payoutDelay + recipientIndex * 120 + index * 80,
                offsetX: ((index % 5) - 2) * 13,
                offsetY: (index % 3) * 7,
                turn: ((index % 5) - 2) * 48,
              }));
            })
          : [];
      if (wagerFlights.length > 0 || payoutFlights.length > 0)
        setChipFlights((flights) => [
          ...flights,
          ...wagerFlights,
          ...payoutFlights,
        ]);
    }
    const upsertHand = (snapshot: HandHistory) =>
      setHands((prev) => {
        const index = prev.findIndex(
          (hand) => hand.handNumber === snapshot.handNumber,
        );
        const next =
          index === -1
            ? [...prev, snapshot]
            : prev.map((hand, position) =>
                position === index ? snapshot : hand,
              );
        storeHands(next);
        return next;
      });
    if (handJustCompleted) {
      upsertHand(snapshotOf(next));
      setHandSettling(true);
    }
    if (startsFreshHand) {
      setHandSettling(false);
      setAutoDealing(false);
      if (newMatch) {
        setHands([]);
        storeHands([]);
        setReplayOpen(false);
        setReplayHandNumber(null);
      } else if (previous?.complete) {
        upsertHand(snapshotOf(previous));
      }
    }
    if (!previous) {
      setDisplayedDealer(next.button as 0 | 1);
    } else if (
      previous.hand_number !== next.hand_number &&
      previous.button !== next.button
    ) {
      const from = previous.button as 0 | 1;
      const to = next.button as 0 | 1;
      if (reducedMotion) {
        setDisplayedDealer(to);
      } else {
        setDisplayedDealer(from);
        setDealerFlight({ id: ++dealerFlightId.current, from, to });
      }
    }
    gameRef.current = next;
    setGame(next);
    setRaiseTo((previous) =>
      boundedRaise(
        previous,
        next.legal_actions,
        next.legal_actions.raise_min ?? previous,
      ),
    );
  }, []);

  const playAgentTurns = useCallback(
    async (initialGame: GameState) => {
      if (
        initialGame.complete ||
        initialGame.current_player !== 1 ||
        agentTurnRunning.current
      )
        return;

      agentTurnRunning.current = true;
      setAgentThinking(true);
      try {
        let current = initialGame;
        while (!current.complete && current.current_player === 1) {
          const thinkTime =
            AGENT_THINK_MIN_MS +
            Math.random() * (AGENT_THINK_MAX_MS - AGENT_THINK_MIN_MS);
          await new Promise<void>((resolve) => {
            agentDelayTimer.current = window.setTimeout(() => {
              agentDelayTimer.current = null;
              resolve();
            }, thinkTime);
          });
          current = await api.agentAction();
          acceptGame(current);
        }
      } finally {
        agentTurnRunning.current = false;
        setAgentThinking(false);
      }
    },
    [acceptGame],
  );

  const refresh = useCallback(async () => {
    try {
      const [currentGame, currentTraining] = await Promise.all([
        api.getGame(),
        api.trainingStatus(),
      ]);
      acceptGame(currentGame);
      setTraining(currentTraining);
      if (
        !currentGame.complete &&
        currentGame.current_player === 1 &&
        !agentTurnRunning.current
      ) {
        setBusy(true);
        try {
          await playAgentTurns(currentGame);
        } finally {
          setBusy(false);
        }
      }
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Cannot reach the Python server.",
      );
    }
  }, [acceptGame, playAgentTurns]);

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    if (!training?.running) return;
    const interval = window.setInterval(() => void refresh(), 700);
    return () => window.clearInterval(interval);
  }, [refresh, training?.running]);

  const sendAction = async (action: string) => {
    if (!game) return;
    void sound.unlock();
    setBusy(true);
    try {
      const amount =
        action === "raise"
          ? boundedRaise(raiseTo, game.legal_actions, raiseTo)
          : undefined;
      if (amount !== undefined) setRaiseTo(amount);
      const next = await api.action(action, amount);
      acceptGame(next);
      await playAgentTurns(next);
      setMessage("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Action failed.");
    } finally {
      setBusy(false);
    }
  };

  const deal = useCallback(
    async (kind: "new" | "next") => {
      cancelAutoDeal();
      void sound.unlock();
      setBusy(true);
      try {
        const next = kind === "new" ? await api.newGame() : await api.nextHand();
        acceptGame(next, kind === "new");
        await playAgentTurns(next);
        setMessage("");
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Unable to deal.");
      } finally {
        setBusy(false);
      }
    },
    [acceptGame, cancelAutoDeal, playAgentTurns],
  );

  const applyGameSettings = async (settings: GameSettings) => {
    cancelAutoDeal();
    void sound.unlock();
    setBusy(true);
    try {
      const next = await api.updateGameSettings(settings);
      acceptGame(next, true);
      await playAgentTurns(next);
      setMessage(
        `New ${settings.small_blind.toLocaleString()}/${settings.big_blind.toLocaleString()} table started with ${settings.initial_stack.toLocaleString()}-chip stacks.`,
      );
      setSettingsOpen(false);
    } finally {
      setBusy(false);
    }
  };

  const reloadGameCash = async (reload: CashReloadRequest) => {
    cancelAutoDeal();
    setBusy(true);
    try {
      acceptGame(await api.reloadCash(reload));
      setMessage("Cash reloaded. Deal the next hand when ready.");
    } finally {
      setBusy(false);
    }
  };

  const startTraining = async () => {
    const safeEpisodes = clamp(
      Number.isFinite(episodes) ? Math.trunc(episodes) : MIN_EPISODES,
      MIN_EPISODES,
      MAX_EPISODES,
    );
    setEpisodes(safeEpisodes);
    setBusy(true);
    try {
      setTraining(await api.train(safeEpisodes));
      setMessage("");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not start training.",
      );
    } finally {
      setBusy(false);
    }
  };

  const reloadLastModel = async () => {
    if (
      !window.confirm(
        "Reload the latest saved checkpoint? Any unsaved in-memory strategy will be discarded.",
      )
    )
      return;
    setBusy(true);
    try {
      const result = await api.reloadLastModel();
      setTraining(result.status);
      const label =
        result.agent === "GpuBlueprintAgent" ? "GPU blueprint" : "CPU blueprint";
      const iterationSuffix =
        result.iteration !== null
          ? ` at iteration ${result.iteration.toLocaleString()}`
          : "";
      setMessage(`Now serving ${label}${iterationSuffix} (${result.source}).`);
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Could not reload the saved checkpoint.",
      );
    } finally {
      setBusy(false);
    }
  };

  const queryCurrentChampion = async () => {
    setChampionOpen(true);
    setChampionBusy(true);
    setChampionError("");
    setChampionResult(null);
    try {
      setChampionResult(await api.queryChampion({ current: true }));
    } catch (error) {
      setChampionError(
        error instanceof Error ? error.message : "Could not query the champion.",
      );
    } finally {
      setChampionBusy(false);
    }
  };

  const openChampionLab = () => {
    setChampionError("");
    setChampionResult(null);
    setChampionOpen(true);
  };

  useEffect(() => {
    if (
      !game?.complete ||
      busy ||
      autoDealTimer.current !== null ||
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
    )
      return;
    setAutoDealing(true);
    autoDealTimer.current = window.setTimeout(() => {
      autoDealTimer.current = null;
      void deal("next");
    }, 2_800);
    return () => {
      if (autoDealTimer.current !== null) {
        window.clearTimeout(autoDealTimer.current);
        autoDealTimer.current = null;
      }
    };
  }, [busy, deal, game?.complete, game?.hand_number]);

  useEffect(
    () => () => {
      if (autoDealTimer.current !== null)
        window.clearTimeout(autoDealTimer.current);
      if (agentDelayTimer.current !== null)
        window.clearTimeout(agentDelayTimer.current);
    },
    [],
  );

  const openReplay = useCallback(
    (handNumber?: number) => {
      const target =
        handNumber ?? hands[hands.length - 1]?.handNumber ?? null;
      const hand = hands.find((entry) => entry.handNumber === target);
      if (!hand) return;
      setReplayHandNumber(hand.handNumber);
      setReplayStep(hand.entries.length);
      setReplayPlaying(false);
      setReplayOpen(true);
    },
    [hands],
  );

  useEffect(() => {
    if (!replayPlaying || !replayOpen) return;
    const hand = hands.find((entry) => entry.handNumber === replayHandNumber);
    if (!hand) return;
    if (replayStep >= hand.entries.length) {
      setReplayPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => setReplayStep((step) => step + 1),
      1_100,
    );
    return () => window.clearTimeout(timer);
  }, [replayPlaying, replayOpen, replayHandNumber, replayStep, hands]);

  useEffect(() => {
    const controls = betControlsRef.current;
    if (!controls) return;

    const handleBetWheel = (event: WheelEvent) => {
      const target = event.target;
      const overBetControl =
        target instanceof Element &&
        target.matches(".table-bet-input, .table-bet-slider");
      const legal = game?.legal_actions;
      const canAdjust =
        overBetControl &&
        game?.current_player === 0 &&
        !game.complete &&
        !busy &&
        legal?.raise &&
        event.deltaY !== 0;
      if (!canAdjust || !legal) return;

      event.preventDefault();
      const direction = event.deltaY < 0 ? 1 : -1;
      setRaiseTo((value) =>
        boundedRaise(value + direction * 10, legal, value),
      );
    };

    controls.addEventListener("wheel", handleBetWheel, { passive: false });
    return () => controls.removeEventListener("wheel", handleBetWheel);
  }, [busy, game]);

  const statusLine = useMemo(() => {
    if (!game) return "Loading game state…";
    if (game.complete) return game.result ?? "Hand complete.";
    if (agentThinking) return "Agent is thinking…";
    return game.current_player === 0 ? "Your action." : "Agent is acting.";
  }, [agentThinking, game]);

  if (!game)
    return (
      <main className="loading">
        <p>{message}</p>
      </main>
    );

  const latestHand = hands.length ? hands[hands.length - 1] : null;
  const legal = game.legal_actions;
  const canAct = game.current_player === 0 && !game.complete && !busy;
  const potDisplay = game.complete ? game.last_pot : game.pot;
  const trainingProgress = Math.round((training?.progress ?? 0) * 100);
  const trainingRate =
    training && training.iterations_per_second > 0
      ? `${training.iterations_per_second.toFixed(1)}/s`
      : "—";
  const servingModel = training?.serving_model;
  const selectedDepth =
    servingModel?.selected_depth_bb === null ||
    servingModel?.selected_depth_bb === undefined
      ? "—"
      : `${servingModel.selected_depth_bb.toLocaleString()} BB`;
  const servingIteration =
    servingModel?.iteration === null || servingModel?.iteration === undefined
      ? "—"
      : servingModel.iteration.toLocaleString();
  const availableDepths = servingModel?.available_depths.length
    ? servingModel.available_depths
        .map((model) => model.depth_bb.toLocaleString())
        .join(" / ") + " BB"
    : "—";
  const searchEnabled =
    servingModel?.search_enabled ?? training?.river_search ?? false;
  const searchIterations = servingModel?.search_iterations;
  const dealerSeat = displayedDealer ?? (game.button as 0 | 1);
  const preflop = game.street.toLowerCase() === "preflop";
  const bigBlind = game.settings?.big_blind ?? bigBlindFromHistory(game.history);
  const currentBet = Math.max(...game.round_bets);
  const callAmount = legal.to_call ?? game.to_call;
  const chipLead = game.stacks[0] - game.stacks[1];
  const chipLeadLabel =
    chipLead === 0 ? "Stacks even" : chipLead > 0 ? "You ahead" : "Agent ahead";
  const sessionStats = game.session_stats;
  const heroStats = sessionStats.players[0];
  const agentStats = sessionStats.players[1];
  const heroBuyIn = heroStats.total_buy_in ?? game.settings?.initial_stack ?? 0;
  const agentBuyIn = agentStats.total_buy_in ?? game.settings?.initial_stack ?? 0;
  const completedHudHands = hands
    .filter((hand) => hand.result !== null)
    .slice(-sessionStats.hands_completed);
  const derivedHudStats = derivePlayerHudStats(completedHudHands);
  const decidedHands = sessionStats.hands_completed - sessionStats.split_pots;
  const heroWinRate =
    decidedHands > 0
      ? Math.round((heroStats.hand_wins / decidedHands) * 100)
      : null;
  const averagePot =
    sessionStats.hands_completed > 0
      ? Math.round(sessionStats.total_pot / sessionStats.hands_completed)
      : null;
  const heroAggression =
    heroStats.aggression === null ? "—" : `${heroStats.aggression}%`;
  const sizedRaisePresets = (
    preflop
      ? [
          { label: "2 BB", target: bigBlind * 2 },
          { label: "2.5 BB", target: bigBlind * 2.5 },
          { label: "3 BB", target: bigBlind * 3 },
        ]
      : [
          { label: "33%", target: currentBet + game.pot * 0.33 },
          { label: "50%", target: currentBet + game.pot * 0.5 },
          { label: "70%", target: currentBet + game.pot * 0.7 },
        ]
  )
    .map((preset) => ({
      ...preset,
      target: boundedRaise(Math.round(preset.target), legal, raiseTo),
    }))
    .filter(
      (preset, index, presets) =>
        presets.findIndex((candidate) => candidate.target === preset.target) ===
        index,
    );
  const quickRaisePresets =
    legal.raise_max === undefined
      ? sizedRaisePresets
      : [
          ...[
            ...sizedRaisePresets,
            {
              label: "Pot",
              target: boundedRaise(
                currentBet + game.pot + callAmount,
                legal,
                raiseTo,
              ),
            },
          ]
            .filter(
              (preset, index, presets) =>
                preset.target !== legal.raise_max &&
                presets.findIndex(
                  (candidate) => candidate.target === preset.target,
                ) === index,
            ),
          { label: "Max", target: legal.raise_max },
        ];

  const replayHand =
    hands.find((entry) => entry.handNumber === replayHandNumber) ?? null;
  const replayIndex = replayHand
    ? hands.findIndex((entry) => entry.handNumber === replayHand.handNumber)
    : -1;
  const replayTotalSteps = replayHand ? replayHand.entries.length : 0;
  const replayClampedStep = Math.min(replayStep, replayTotalSteps);
  const replayPlayed = replayHand
    ? replayHand.entries.slice(0, replayClampedStep)
    : [];
  const replayBoard = replayHand
    ? replayHand.community.slice(0, boardCountAtStep(replayPlayed))
    : [];
  const replayCurrentEntry =
    replayClampedStep > 0 ? replayPlayed[replayClampedStep - 1] : null;
  const replayFinished =
    replayHand !== null && replayClampedStep >= replayTotalSteps;
  const replayOutcome = replayHand ? handOutcome(replayHand) : null;
  const gotoReplayStep = (step: number) => {
    setReplayPlaying(false);
    setReplayStep(clamp(step, 0, replayTotalSteps));
  };
  const gotoReplayHand = (index: number) => {
    const target = hands[index];
    if (!target) return;
    setReplayPlaying(false);
    setReplayHandNumber(target.handNumber);
    setReplayStep(target.entries.length);
  };
  const toggleReplayPlay = () => {
    if (!replayHand) return;
    if (replayFinished) setReplayStep(0);
    setReplayPlaying((playing) => !playing);
  };
  const updateSoundSettings = (next: SoundSettings) => {
    sound.configure(next);
    setSoundSettings(next);
  };

  return (
    <main className="app-shell">
      <header className="masthead">
        <div className="brand-lockup">
          <p className="eyebrow">SELF-PLAY LAB / TABLE 01</p>
          <h1>TEXT HOLD’EM</h1>
        </div>
        <div className="masthead-hand-summary" aria-label="Current hand summary">
          <span>
            <small>Hand</small>
            <strong>#{game.hand_number}</strong>
          </span>
          <span>
            <small>Street</small>
            <strong>{game.street}</strong>
          </span>
          <span>
            <small>{game.complete ? "Last pot" : "Pot"}</small>
            <strong>{potDisplay.toLocaleString()}</strong>
          </span>
        </div>
        <div className="masthead-controls">
          <section
            className="top-action-bar"
            aria-label="Your action"
            aria-live="polite"
          >
            <div className="top-action-status">
              <span className={game.complete ? "signal done" : "signal"} />
              {statusLine}
            </div>
            {!game.complete ? (
              <div className="top-action-bets">
                <span>
                  Call <strong>{game.to_call.toLocaleString()}</strong>
                </span>
                <span>
                  Bet{" "}
                  <strong>
                    {Math.max(...game.round_bets).toLocaleString()}
                  </strong>
                </span>
              </div>
            ) : autoDealing ? (
              <span className="top-action-auto">Next hand dealing…</span>
            ) : (
              <button
                className="accent top-action-next"
                onClick={() => void deal("next")}
                disabled={busy}
              >
                Deal next hand →
              </button>
            )}
            <button
              className="top-action-history"
              onClick={() => openReplay()}
              disabled={!latestHand}
            >
              Hand replay{latestHand ? ` (${hands.length})` : ""}
            </button>
            <button
              className="top-action-champion"
              onClick={() =>
                canAct ? void queryCurrentChampion() : openChampionLab()
              }
              disabled={championBusy}
            >
              {canAct ? "Ask champion" : "Hand lab"}
            </button>
            <button
              className="top-action-settings"
              type="button"
              onClick={() => setSettingsOpen(true)}
              disabled={busy}
            >
              Settings
            </button>
            <div className="sound-controls" aria-label="Sound controls">
              <button
                className="top-action-sound"
                type="button"
                onClick={() => {
                  const next = {
                    ...soundSettings,
                    enabled: !soundSettings.enabled,
                  };
                  updateSoundSettings(next);
                  if (next.enabled) void sound.unlock();
                }}
                aria-label={soundSettings.enabled ? "Mute sounds" : "Enable sounds"}
                aria-pressed={soundSettings.enabled}
                title={soundSettings.enabled ? "Mute sounds" : "Enable sounds"}
              >
                {soundSettings.enabled ? "Sound on" : "Sound off"}
              </button>
              <label>
                <span className="sr-only">Sound volume</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={soundSettings.volume}
                  onChange={(event) =>
                    updateSoundSettings({
                      ...soundSettings,
                      volume: Number(event.target.value),
                    })
                  }
                  aria-label="Sound volume"
                />
              </label>
            </div>
            {message && (
              <p className="top-action-message" role="alert">
                {message}
              </p>
            )}
          </section>
          <div className="table-identity">
            <span>HOLD’EM</span>
            <small>HEADS UP · NLH</small>
          </div>
        </div>
      </header>

      <div className="game-workspace">
        <section
          className={`poker-table ${handSettling ? "hand-settling" : ""} ${autoDealing ? "auto-dealing" : ""}`}
          aria-label="Three-dimensional heads-up poker table"
        >
          <div className="table-shadow" />
          <div className="table-base" />
          <div className="table-rail" />
          <div className="table-felt">
            <div className="felt-grain" />
          </div>
          <button
            className="table-new-match"
            onClick={() => void deal("new")}
            disabled={busy}
          >
            New match
          </button>
          <div className="seat opponent-seat">
            <div className="player-plaque player-plaque-hud">
              <span className="seat-indicator">02</span>
              <div className="player-info">
                <p>AGENT {game.button === 1 && <em>BUTTON</em>}</p>
                <div className="player-stack">
                  <strong>{game.stacks[1].toLocaleString()}</strong>
                  <small>chips</small>
                  <span
                    className="player-buy-in"
                    title="Total chips bought during this session"
                  >
                    <small>Buy-in</small>
                    <b>{agentBuyIn.toLocaleString()}</b>
                  </span>
                </div>
                <PlayerHud
                  stats={agentStats}
                  hands={sessionStats.hands_completed}
                  derived={derivedHudStats[1]}
                  historyHands={completedHudHands.length}
                  player="Agent"
                />
              </div>
            </div>
            <Cards
              key={`opponent-${game.hand_number}`}
              cards={game.opponent_cards}
              hidden={!game.complete}
              className="opponent-cards"
            />
          </div>

          {game.round_bets[1] > 0 && (
            <ChipStack
              amount={game.round_bets[1]}
              className="wager opponent-wager"
            />
          )}
          {game.round_bets[0] > 0 && (
            <ChipStack
              amount={game.round_bets[0]}
              className="wager hero-wager"
            />
          )}

          <div className="pot-zone">
            <div className="pot-display">
              <span>{game.complete ? "LAST POT" : "POT"}</span>
              <ChipStack amount={potDisplay} />
            </div>
            <p className="street">
              {game.street.toUpperCase()} · HAND #{game.hand_number}
            </p>
          </div>

          <div className="board-zone">
            {game.community.length > 0 ? (
              <Cards cards={game.community} className="community-cards" />
            ) : (
              <div className="board-awaiting">FLOP · TURN · RIVER</div>
            )}
          </div>

          <div className="chip-flight-layer" aria-hidden="true">
            {chipFlights.map((flight) => (
              <div
                className={`chip-flight chip-flight-${flight.motion}-${flight.target}`}
                key={flight.id}
                style={
                  {
                    animationDelay: `${flight.delay}ms`,
                    "--chip-offset-x": `${flight.offsetX}px`,
                    "--chip-offset-y": `${flight.offsetY}px`,
                    "--chip-turn": `${flight.turn}deg`,
                  } as CSSProperties
                }
                onAnimationEnd={() =>
                  setChipFlights((flights) =>
                    flights.filter((item) => item.id !== flight.id),
                  )
                }
              >
                <img
                  className="chip-flight-art"
                  src={chipAssetPath(flight.colour)}
                  alt=""
                />
              </div>
            ))}
          </div>

          <div
            className={`dealer-disc dealer-player-${dealerSeat}`}
            aria-label={`Player ${dealerSeat + 1} has the dealer button`}
          >
            <img
              className="dealer-button-art"
              src="/assets/dealer-button.png"
              alt=""
            />
            <b>BTN</b>
          </div>
          {dealerFlight && (
            <div
              className={`dealer-flight dealer-flight-${dealerFlight.from}-to-${dealerFlight.to}`}
              aria-hidden="true"
              onAnimationEnd={() => {
                setDisplayedDealer(dealerFlight.to);
                setDealerFlight(null);
              }}
            >
              <img
                className="dealer-button-art"
                src="/assets/dealer-button.png"
                alt=""
              />
              <b>BTN</b>
            </div>
          )}

          {!game.complete && (
            <div className="table-actions" aria-label="Poker actions">
              {legal.raise && (
                <div
                  className="table-bet-presets"
                  aria-label="Quick raise sizes"
                >
                  {quickRaisePresets.map((preset) => (
                    <button
                      className={`table-bet-preset ${raiseTo === preset.target ? "active" : ""}`}
                      key={preset.label}
                      type="button"
                      title={`Set raise to ${preset.target.toLocaleString()}`}
                      aria-pressed={raiseTo === preset.target}
                      onClick={() => setRaiseTo(preset.target)}
                      disabled={!canAct}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              )}
              {legal.raise && (
                <div className="table-bet-controls" ref={betControlsRef}>
                  <input
                    className="table-bet-input"
                    type="number"
                    min={legal.raise_min}
                    max={legal.raise_max}
                    step={10}
                    value={raiseTo}
                    aria-label="Raise-to amount"
                    title="Scroll to adjust by 10 chips"
                    onChange={(event) => setRaiseTo(Number(event.target.value))}
                    onBlur={() =>
                      setRaiseTo((value) => boundedRaise(value, legal, value))
                    }
                    disabled={!canAct}
                  />
                  <input
                    className="table-bet-slider"
                    type="range"
                    min={legal.raise_min}
                    max={legal.raise_max}
                    step={1}
                    value={raiseTo}
                    aria-label="Raise-to amount slider"
                    title="Scroll to adjust by 10 chips"
                    onChange={(event) => setRaiseTo(Number(event.target.value))}
                    disabled={!canAct}
                  />
                </div>
              )}
              <div className="table-main-actions">
                {legal.fold && (
                  <button
                    className="table-fold"
                    onClick={() => void sendAction("fold")}
                    disabled={!canAct}
                  >
                    Fold
                  </button>
                )}
                {legal.check && (
                  <button
                    className="table-check"
                    onClick={() => void sendAction("check")}
                    disabled={!canAct}
                  >
                    Check
                  </button>
                )}
                {legal.call && (
                  <button
                    className="table-call"
                    onClick={() => void sendAction("call")}
                    disabled={!canAct}
                  >
                    <span>Call</span>
                    <strong>{legal.to_call?.toLocaleString()}</strong>
                  </button>
                )}
                {legal.raise && (
                  <button
                    className="table-raise-action"
                    onClick={() => void sendAction("raise")}
                    disabled={!canAct}
                  >
                    <span>{currentBet === 0 ? "Bet" : "Raise To"}</span>
                    <strong>{raiseTo.toLocaleString()}</strong>
                  </button>
                )}
              </div>
            </div>
          )}

          <div className="seat hero-seat">
            <Cards
              key={`hero-${game.hand_number}`}
              cards={game.hero_cards}
              className="hero-cards"
            />
            <div className="player-plaque player-plaque-hud">
              <span className="seat-indicator">01</span>
              <div className="player-info">
                <p>YOU {game.button === 0 && <em>BUTTON</em>}</p>
                <div className="player-stack">
                  <strong>{game.stacks[0].toLocaleString()}</strong>
                  <small>chips</small>
                  <span
                    className="player-buy-in"
                    title="Total chips bought during this session"
                  >
                    <small>Buy-in</small>
                    <b>{heroBuyIn.toLocaleString()}</b>
                  </span>
                </div>
                <PlayerHud
                  stats={heroStats}
                  hands={sessionStats.hands_completed}
                  derived={derivedHudStats[0]}
                  historyHands={completedHudHands.length}
                  player="Your"
                />
                <div className="hand-strength" aria-live="polite">
                  <span>Hand strength</span>
                  <strong>{game.hero_hand_strength ?? "—"}</strong>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="match-stats" aria-label="Match statistics">
          <header className="match-stats-header">
            <div>
              <span className="signal" />
              MATCH STATS
            </div>
            <p>SESSION · HAND #{game.hand_number}</p>
          </header>
          <div className="match-scoreboard">
            <div className="match-scoreboard-title">
              <span>Hand record</span>
              <small>
                Match {heroStats.match_wins}—{agentStats.match_wins}
              </small>
            </div>
            <div className="match-scoreboard-duel">
              <div className="match-player hero">
                <span>You</span>
                <strong>{heroStats.hand_wins}</strong>
                <small>hand wins</small>
              </div>
              <div className="match-win-rate">
                <span>Win rate</span>
                <strong>{heroWinRate === null ? "—" : `${heroWinRate}%`}</strong>
                <small>
                  {decidedHands ? `${decidedHands} decided` : "No results"}
                </small>
              </div>
              <div className="match-player agent">
                <span>Agent</span>
                <strong>{agentStats.hand_wins}</strong>
                <small>hand wins</small>
              </div>
            </div>
            <div
              className="match-win-track"
              role="progressbar"
              aria-label="Your hand win rate"
              aria-valuenow={heroWinRate ?? 0}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span style={{ width: `${heroWinRate ?? 0}%` }} />
            </div>
          </div>

          <div className="match-primary-grid">
            <div className="match-primary-stat">
              <span>Hands</span>
              <strong>{sessionStats.hands_completed}</strong>
              <small>{game.complete ? "Settled" : "In progress"}</small>
            </div>
            <div
              className={`match-primary-stat ${chipLead === 0 ? "" : chipLead > 0 ? "hero-lead" : "agent-lead"}`}
            >
              <span>Chip lead</span>
              <strong>
                {chipLead === 0 ? "—" : Math.abs(chipLead).toLocaleString()}
              </strong>
              <small>{chipLeadLabel}</small>
            </div>
            <div className="match-primary-stat">
              <span>Average pot</span>
              <strong>
                {averagePot === null ? "—" : averagePot.toLocaleString()}
              </strong>
              <small>chips contested</small>
            </div>
            <div className="match-primary-stat highlight">
              <span>Biggest pot</span>
              <strong>
                {sessionStats.biggest_pot
                  ? sessionStats.biggest_pot.toLocaleString()
                  : "—"}
              </strong>
              <small>session high</small>
            </div>
          </div>

          <div className="match-secondary-stats">
            <div className="match-secondary-title">Session detail</div>
            <div className="match-secondary-row">
              <span>Showdowns</span>
              <strong>{sessionStats.showdown_hands}</strong>
              <small>
                You {heroStats.showdown_wins} · Agent {agentStats.showdown_wins} ·{" "}
                {sessionStats.split_pots} split
              </small>
            </div>
            <div className="match-secondary-row">
              <span>Fold wins</span>
              <strong>
                {heroStats.fold_wins}—{agentStats.fold_wins}
              </strong>
              <small>You · Agent</small>
            </div>
            <div className="match-secondary-row">
              <span>Playing style</span>
              <strong>{heroStats.vpip}% VPIP</strong>
              <small>
                PFR {heroStats.pfr}% · Agg {heroAggression}
              </small>
            </div>
            <div className="match-secondary-row">
              <span>Total buy-in</span>
              <strong>
                {heroBuyIn.toLocaleString()}—{agentBuyIn.toLocaleString()}
              </strong>
              <small>You · Agent</small>
            </div>
          </div>

        </section>

        <aside className="workspace-sidebar">
          <section className="command-deck">
            <div className="status" aria-live="polite">
              <span className={game.complete ? "signal done" : "signal"} />
              {statusLine}
            </div>
            {!game.complete ? (
              <>
                <div className="betting-summary">
                  <span>
                    To call <strong>{game.to_call.toLocaleString()}</strong>
                  </span>
                  <span>
                    Highest bet{" "}
                    <strong>
                      {Math.max(...game.round_bets).toLocaleString()}
                    </strong>
                  </span>
                </div>
                <p className="table-action-hint">
                  Actions are positioned on the table’s left rail.
                </p>
              </>
            ) : (
              <button
                className="accent next"
                onClick={() => void deal("next")}
                disabled={busy}
              >
                Deal next hand →
              </button>
            )}
            <button
              className="history-button"
              onClick={() => openReplay()}
              disabled={!latestHand}
            >
              Hand replay{latestHand ? ` (${hands.length})` : ""}
            </button>
            <div className="champion-launchers">
              <button
                className="champion-button"
                onClick={() => void queryCurrentChampion()}
                disabled={!canAct || championBusy}
              >
                Ask champion
              </button>
              <button onClick={openChampionLab} disabled={championBusy}>
                Hand lab
              </button>
            </div>
            <button
              className="settings-button"
              type="button"
              onClick={() => setSettingsOpen(true)}
              disabled={busy}
            >
              Game settings
            </button>
            {message && (
              <p className="message" role="alert">
                {message}
              </p>
            )}
          </section>

          <section className="lower-grid single-panel">
            <article className="panel trainer">
              <header className="trainer-header">
                <div>
                  <div className="panel-title">BLUEPRINT STATUS</div>
                  <p className="trainer-kicker">Serving model + CPU trainer</p>
                </div>
                <div
                  className={`trainer-state ${training?.running ? "running" : "ready"}`}
                >
                  <i />
                  <span>{training?.running ? "CPU training" : "CPU idle"}</span>
                </div>
              </header>
              <p className="trainer-summary">
                The table uses the serving model below. These controls train the
                separate CPU Linear MCCFR average-strategy blueprint.
              </p>
              <div className="trainer-overview">
                <div className="trainer-stat">
                  <span>Serving model</span>
                  <strong>{servingAgentLabel(training?.serving_agent)}</strong>
                  <small>model currently making table decisions</small>
                </div>
                <div className="trainer-stat">
                  <span>Selected depth</span>
                  <strong>{selectedDepth}</strong>
                  <small>checkpoint {servingIteration}</small>
                </div>
                <div className="trainer-stat">
                  <span>Available depths</span>
                  <strong>{availableDepths}</strong>
                  <small>
                    {servingModel?.available_depths.length
                      ? `${servingModel.available_depths.length} served checkpoint${servingModel.available_depths.length === 1 ? "" : "s"}`
                      : "metadata available after server restart"}
                  </small>
                </div>
                <div className="trainer-stat">
                  <span>Late-street search</span>
                  <strong>
                    {searchEnabled
                      ? searchIterations
                        ? `${searchIterations.toLocaleString()} iters`
                        : "On"
                      : "Off"}
                  </strong>
                  <small>turn + river re-solving</small>
                </div>
              </div>
              <div className="trainer-controls">
                <label>
                  <span>Iterations</span>
                  <input
                    type="number"
                    min={MIN_EPISODES}
                    max={MAX_EPISODES}
                    value={episodes}
                    onChange={(event) =>
                      setEpisodes(Number(event.target.value))
                    }
                    onBlur={() =>
                      setEpisodes((value) =>
                        clamp(
                          Math.trunc(value) || MIN_EPISODES,
                          MIN_EPISODES,
                          MAX_EPISODES,
                        ),
                      )
                    }
                    disabled={training?.running}
                  />
                </label>
                <button
                  className="accent"
                  onClick={() => void startTraining()}
                  disabled={busy || training?.running}
                >
                  {training?.running ? "Training…" : "Start training"}
                </button>
              </div>
              <div className="trainer-controls model-controls">
                <button
                  onClick={() => void reloadLastModel()}
                  disabled={busy || training?.running}
                >
                  Reload latest checkpoint
                </button>
              </div>
              {training?.running && (
                <>
                  <div className="progress-label">
                    <span>
                      {`MCCFR iterations ${training.completed.toLocaleString()} / ${training.episodes.toLocaleString()}`}
                    </span>
                    <span>{trainingProgress}%</span>
                  </div>
                  <div
                    className="progress-track"
                    role="progressbar"
                    aria-valuenow={trainingProgress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <span style={{ width: `${trainingProgress}%` }} />
                  </div>
                </>
              )}
              <div className="metrics trainer-telemetry">
                <span>CPU trainer: Linear MCCFR</span>
                <span>
                  CPU checkpoint: {(training?.updates ?? 0).toLocaleString()}
                </span>
                <span>
                  CPU infosets: {(training?.parameters ?? 0).toLocaleString()}
                </span>
                <span>
                  CPU rate: {trainingRate} ·{" "}
                  {training?.running ? "latest chunk" : "last recorded"}
                </span>
                <span>
                  Abstraction{" "}
                  {training?.artifacts.abstraction ? "ready" : "missing"}
                </span>
                <span>
                  CPU blueprint {training?.artifacts.blueprint ? "ready" : "missing"}
                </span>
              </div>
              {training?.last_error && (
                <p className="message">Training error: {training.last_error}</p>
              )}
            </article>
          </section>
        </aside>
      </div>
      {settingsOpen && (
        <GameSettingsDialog
          game={game}
          busy={busy}
          onApply={applyGameSettings}
          onReload={reloadGameCash}
          onClose={() => setSettingsOpen(false)}
        />
      )}
      {championOpen && (
        <ChampionLab
          canUseCurrent={canAct}
          initialBusy={championBusy}
          initialError={championError}
          initialResult={championResult}
          onClose={() => setChampionOpen(false)}
          onUseCurrent={() => void queryCurrentChampion()}
        />
      )}
      {replayOpen && replayHand && (
        <div
          className="hand-history-backdrop"
          role="presentation"
          onMouseDown={() => setReplayOpen(false)}
        >
          <section
            className="replay-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="replay-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="replay-header">
              <div>
                <p className="eyebrow">HAND REPLAY</p>
                <h2 id="replay-title">Hand #{replayHand.handNumber}</h2>
              </div>
              <div className="replay-hand-nav">
                <button
                  className="ghost"
                  onClick={() => gotoReplayHand(replayIndex - 1)}
                  disabled={replayIndex <= 0}
                  aria-label="Older hand"
                >
                  ‹ Older
                </button>
                <span className="replay-hand-count">
                  {replayIndex + 1} / {hands.length}
                </span>
                <button
                  className="ghost"
                  onClick={() => gotoReplayHand(replayIndex + 1)}
                  disabled={replayIndex >= hands.length - 1}
                  aria-label="Newer hand"
                >
                  Newer ›
                </button>
              </div>
              <button className="ghost" onClick={() => setReplayOpen(false)}>
                Close
              </button>
            </header>

            <div className="replay-body">
              <aside className="replay-hand-list" aria-label="All hands">
                {[...hands].reverse().map((hand) => {
                  const outcome = handOutcome(hand);
                  return (
                    <button
                      className={`replay-hand-item ${outcome.tone} ${
                        hand.handNumber === replayHand.handNumber ? "active" : ""
                      }`}
                      key={hand.handNumber}
                      onClick={() =>
                        gotoReplayHand(
                          hands.findIndex(
                            (entry) => entry.handNumber === hand.handNumber,
                          ),
                        )
                      }
                    >
                      <span className="replay-hand-item-no">
                        #{hand.handNumber}
                      </span>
                      <span className="replay-hand-item-outcome">
                        {outcome.label}
                      </span>
                    </button>
                  );
                })}
              </aside>

              <div className="replay-stage">
                <div className="replay-seat replay-seat-opponent">
                  <span className="replay-seat-label">AGENT</span>
                  <Cards
                    cards={replayHand.opponentCards}
                    className="replay-cards"
                  />
                </div>

                <div className="replay-board">
                  {replayBoard.length > 0 ? (
                    <Cards cards={replayBoard} className="replay-cards" />
                  ) : (
                    <div className="replay-board-empty">PRE-FLOP</div>
                  )}
                </div>

                <div className="replay-seat replay-seat-hero">
                  <Cards cards={replayHand.heroCards} className="replay-cards" />
                  <span className="replay-seat-label">YOU</span>
                </div>

                <div
                  className={`replay-callout ${
                    replayCurrentEntry ? actionPopupTone(replayCurrentEntry) : ""
                  }`}
                  aria-live="polite"
                >
                  {replayCurrentEntry
                    ? actionPopupText(replayCurrentEntry)
                    : "Ready to deal"}
                </div>
              </div>

              <ol className="replay-log" aria-label="Action timeline">
                {replayHand.entries.map((entry, index) => (
                  <li
                    className={`replay-log-entry ${actionPopupTone(entry)} ${
                      index < replayClampedStep ? "played" : "future"
                    } ${index === replayClampedStep - 1 ? "current" : ""}`}
                    key={`${entry}-${index}`}
                  >
                    <button onClick={() => gotoReplayStep(index + 1)}>
                      {actionPopupText(entry)}
                    </button>
                  </li>
                ))}
              </ol>
            </div>

            <div className="replay-controls">
              <div className="replay-buttons">
                <button
                  onClick={() => gotoReplayStep(0)}
                  disabled={replayClampedStep === 0}
                  aria-label="Jump to start"
                >
                  ⏮
                </button>
                <button
                  onClick={() => gotoReplayStep(replayClampedStep - 1)}
                  disabled={replayClampedStep === 0}
                  aria-label="Step back"
                >
                  ◀
                </button>
                <button
                  className="accent replay-play"
                  onClick={toggleReplayPlay}
                  aria-label={replayPlaying ? "Pause" : "Play"}
                >
                  {replayPlaying ? "❚❚ Pause" : replayFinished ? "↺ Replay" : "▶ Play"}
                </button>
                <button
                  onClick={() => gotoReplayStep(replayClampedStep + 1)}
                  disabled={replayFinished}
                  aria-label="Step forward"
                >
                  ▶
                </button>
                <button
                  onClick={() => gotoReplayStep(replayTotalSteps)}
                  disabled={replayFinished}
                  aria-label="Jump to end"
                >
                  ⏭
                </button>
              </div>
              <input
                className="replay-scrubber"
                type="range"
                min={0}
                max={replayTotalSteps}
                value={replayClampedStep}
                onChange={(event) => gotoReplayStep(Number(event.target.value))}
                aria-label="Replay position"
              />
              <div className="replay-progress">
                <span>
                  Step {replayClampedStep} / {replayTotalSteps}
                </span>
                {replayFinished && replayOutcome && (
                  <span className={`replay-result ${replayOutcome.tone}`}>
                    {replayHand.result
                      ? actionPopupText(replayHand.result)
                      : replayOutcome.label}
                  </span>
                )}
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
