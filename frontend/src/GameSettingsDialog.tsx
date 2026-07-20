import { useEffect, useMemo, useState, type FormEvent } from "react";
import "./GameSettingsDialog.css";
import type { CashReloadRequest, GameSettings, GameState } from "./types";

type GameSettingsDialogProps = {
  game: GameState;
  busy: boolean;
  onApply: (settings: GameSettings) => Promise<void>;
  onReload: (reload: CashReloadRequest) => Promise<void>;
  onClose: () => void;
};

const DEFAULT_SETTINGS: GameSettings = {
  initial_stack: 2_000,
  small_blind: 10,
  big_blind: 20,
};

const wholeNumber = (value: number) =>
  Number.isFinite(value) ? Math.trunc(value) : 0;

export default function GameSettingsDialog({
  game,
  busy,
  onApply,
  onReload,
  onClose,
}: GameSettingsDialogProps) {
  const current = game.settings ?? DEFAULT_SETTINGS;
  const [initialStack, setInitialStack] = useState(current.initial_stack);
  const [smallBlind, setSmallBlind] = useState(current.small_blind);
  const [bigBlind, setBigBlind] = useState(current.big_blind);
  const [reloadPlayer, setReloadPlayer] =
    useState<CashReloadRequest["player"]>(0);
  const [reloadAmount, setReloadAmount] = useState(current.initial_stack);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const depth = useMemo(
    () => (bigBlind > 0 ? initialStack / bigBlind : 0),
    [bigBlind, initialStack],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !submitting && !busy) onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [busy, onClose, submitting]);

  const validateSettings = (): GameSettings => {
    const settings = {
      initial_stack: wholeNumber(initialStack),
      small_blind: wholeNumber(smallBlind),
      big_blind: wholeNumber(bigBlind),
    };
    if (settings.small_blind < 1 || settings.big_blind < 2) {
      throw new Error("Blinds must be positive whole-chip amounts.");
    }
    if (settings.big_blind !== settings.small_blind * 2) {
      throw new Error("Use standard heads-up blinds: the big blind must be twice the small blind.");
    }
    if (settings.initial_stack < settings.big_blind) {
      throw new Error("Starting stack must be at least one big blind.");
    }
    if (Math.max(settings.initial_stack, settings.small_blind, settings.big_blind) > 1_000_000_000) {
      throw new Error("Chip values cannot exceed 1,000,000,000.");
    }
    return settings;
  };

  const applySettings = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setNotice("");
    try {
      const settings = validateSettings();
      setSubmitting(true);
      await onApply(settings);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not apply the table setup.");
    } finally {
      setSubmitting(false);
    }
  };

  const reloadCash = async () => {
    setError("");
    setNotice("");
    const amount = wholeNumber(reloadAmount);
    if (amount < 1 || amount > 1_000_000_000) {
      setError("Reload amount must be between 1 and 1,000,000,000 chips.");
      return;
    }
    try {
      setSubmitting(true);
      await onReload({ player: reloadPlayer, amount });
      const target = reloadPlayer === 0 ? "your stack" : reloadPlayer === 1 ? "the agent stack" : "both stacks";
      setNotice(`Added ${amount.toLocaleString()} chips to ${target}.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not reload cash.");
    } finally {
      setSubmitting(false);
    }
  };

  const unavailable = busy || submitting;

  return (
    <div
      className="game-settings-backdrop"
      role="presentation"
      onMouseDown={() => {
        if (!unavailable) onClose();
      }}
    >
      <section
        className="game-settings-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="game-settings-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="game-settings-header">
          <div>
            <p className="eyebrow">LIVE TABLE · PLAY ONLY</p>
            <h2 id="game-settings-title">Game settings</h2>
          </div>
          <button className="ghost" type="button" onClick={onClose} disabled={unavailable}>
            Close
          </button>
        </header>

        <div className="game-settings-body">
          <form className="game-setup-card" onSubmit={applySettings}>
            <div className="game-settings-section-heading">
              <div>
                <span>Table setup</span>
                <h3>Blinds and buy-in</h3>
              </div>
              <strong>{Number.isFinite(depth) ? depth.toFixed(depth % 1 ? 1 : 0) : "0"} BB</strong>
            </div>

            <div className="game-settings-fields">
              <label>
                <span>Small blind</span>
                <input
                  autoFocus
                  type="number"
                  min="1"
                  max="1000000000"
                  step="1"
                  value={smallBlind}
                  onChange={(event) => setSmallBlind(Number(event.target.value))}
                  disabled={unavailable}
                />
              </label>
              <label>
                <span>Big blind</span>
                <input
                  type="number"
                  min="2"
                  max="1000000000"
                  step="1"
                  value={bigBlind}
                  onChange={(event) => setBigBlind(Number(event.target.value))}
                  disabled={unavailable}
                />
              </label>
              <label className="wide">
                <span>Starting stack · each player</span>
                <input
                  type="number"
                  min="1"
                  max="1000000000"
                  step="1"
                  value={initialStack}
                  onChange={(event) => setInitialStack(Number(event.target.value))}
                  disabled={unavailable}
                />
              </label>
            </div>

            <div className="game-settings-presets" aria-label="Stack depth presets">
              {[50, 100, 200].map((stackBb) => (
                <button
                  type="button"
                  key={stackBb}
                  onClick={() => setInitialStack(bigBlind * stackBb)}
                  disabled={unavailable || bigBlind < 1}
                >
                  {stackBb} BB
                </button>
              ))}
            </div>

            <p className="game-settings-explainer">
              Applying setup starts a fresh match and clears match statistics. Training keeps its own solver configuration; live play only selects the nearest available blueprint for this stack depth.
            </p>

            <button className="accent game-settings-apply" type="submit" disabled={unavailable}>
              {submitting ? "Applying…" : "Apply & start new match"}
            </button>
          </form>

          <section className="cash-reload-card" aria-labelledby="cash-reload-title">
            <div className="game-settings-section-heading">
              <div>
                <span>Cash table</span>
                <h3 id="cash-reload-title">Reload chips</h3>
              </div>
            </div>

            <div className="cash-stack-summary">
              <div>
                <span>You</span>
                <strong>{game.stacks[0].toLocaleString()}</strong>
              </div>
              <div>
                <span>Agent</span>
                <strong>{game.stacks[1].toLocaleString()}</strong>
              </div>
            </div>

            <label>
              <span>Add chips to</span>
              <select
                value={reloadPlayer}
                onChange={(event) => {
                  const value = event.target.value;
                  setReloadPlayer(value === "both" ? "both" : value === "1" ? 1 : 0);
                }}
                disabled={unavailable || !game.complete}
              >
                <option value="0">You</option>
                <option value="1">Agent</option>
                <option value="both">Both players</option>
              </select>
            </label>
            <label>
              <span>Reload amount</span>
              <input
                type="number"
                min="1"
                max="1000000000"
                step="1"
                value={reloadAmount}
                onChange={(event) => setReloadAmount(Number(event.target.value))}
                disabled={unavailable || !game.complete}
              />
            </label>
            <button
              type="button"
              onClick={() => void reloadCash()}
              disabled={unavailable || !game.complete}
            >
              {submitting ? "Reloading…" : "Add cash"}
            </button>
            <p className="game-settings-explainer">
              {game.complete
                ? "Reloading adds chips and preserves this match's results and winnings."
                : "Finish the current hand before adding chips."}
            </p>
          </section>
        </div>

        {(error || notice) && (
          <p className={`game-settings-feedback ${error ? "error" : "success"}`} role={error ? "alert" : "status"}>
            {error || notice}
          </p>
        )}
      </section>
    </div>
  );
}
