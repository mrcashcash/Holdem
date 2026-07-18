export type SoundEffect = "deal" | "chips" | "check" | "fold" | "win";

export type SoundSettings = {
  enabled: boolean;
  volume: number;
};

const STORAGE_KEY = "holdem.soundSettings.v1";
export const DEFAULT_SOUND_SETTINGS: SoundSettings = {
  enabled: true,
  volume: 0.45,
};

const clampVolume = (value: number) => Math.min(Math.max(value, 0), 1);

export const loadSoundSettings = (): SoundSettings => {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) return DEFAULT_SOUND_SETTINGS;
    const parsed = JSON.parse(stored) as Partial<SoundSettings>;
    return {
      enabled:
        typeof parsed.enabled === "boolean"
          ? parsed.enabled
          : DEFAULT_SOUND_SETTINGS.enabled,
      volume:
        typeof parsed.volume === "number"
          ? clampVolume(parsed.volume)
          : DEFAULT_SOUND_SETTINGS.volume,
    };
  } catch {
    return DEFAULT_SOUND_SETTINGS;
  }
};

export const storeSoundSettings = (settings: SoundSettings) => {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable — sound preferences apply for this session only */
  }
};

class SoundManager {
  private context: AudioContext | null = null;
  private master: GainNode | null = null;
  private settings = DEFAULT_SOUND_SETTINGS;

  configure(settings: SoundSettings) {
    this.settings = settings;
    if (this.master)
      this.master.gain.setTargetAtTime(
        settings.enabled ? settings.volume : 0,
        this.context?.currentTime ?? 0,
        0.015,
      );
  }

  async unlock() {
    if (this.context) {
      if (this.context.state === "suspended") await this.context.resume();
      return;
    }

    const AudioContextConstructor =
      window.AudioContext ??
      (window as typeof window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!AudioContextConstructor) return;

    this.context = new AudioContextConstructor();
    this.master = this.context.createGain();
    this.master.gain.value = this.settings.enabled ? this.settings.volume : 0;
    this.master.connect(this.context.destination);
    if (this.context.state === "suspended") await this.context.resume();
  }

  play(effect: SoundEffect) {
    const context = this.context;
    if (!context || context.state !== "running" || !this.settings.enabled) return;

    switch (effect) {
      case "deal":
        [0, 0.055, 0.11].forEach((offset) =>
          this.noise(offset, 0.045, 1_850, 0.1),
        );
        break;
      case "chips":
        this.tone(1_250, 0.065, "triangle", 0, 0.14);
        this.tone(1_700, 0.04, "sine", 0.035, 0.075);
        break;
      case "check":
        this.tone(720, 0.06, "sine", 0, 0.09);
        break;
      case "fold":
        this.noise(0, 0.13, 520, 0.13);
        break;
      case "win":
        [660, 830, 990].forEach((frequency, index) =>
          this.tone(frequency, 0.15, "triangle", index * 0.095, 0.12),
        );
        break;
    }
  }

  private tone(
    frequency: number,
    duration: number,
    type: OscillatorType,
    offset: number,
    level: number,
  ) {
    if (!this.context || !this.master) return;
    const start = this.context.currentTime + offset;
    const gain = this.context.createGain();
    const oscillator = this.context.createOscillator();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(frequency, start);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(level, start + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    oscillator.connect(gain);
    gain.connect(this.master);
    oscillator.start(start);
    oscillator.stop(start + duration + 0.02);
  }

  private noise(offset: number, duration: number, cutoff: number, level: number) {
    if (!this.context || !this.master) return;
    const start = this.context.currentTime + offset;
    const frameCount = Math.ceil(this.context.sampleRate * duration);
    const buffer = this.context.createBuffer(1, frameCount, this.context.sampleRate);
    const data = buffer.getChannelData(0);
    for (let index = 0; index < data.length; index += 1)
      data[index] = Math.random() * 2 - 1;

    const source = this.context.createBufferSource();
    const filter = this.context.createBiquadFilter();
    const gain = this.context.createGain();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(cutoff, start);
    gain.gain.setValueAtTime(level, start);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    source.buffer = buffer;
    source.connect(filter);
    filter.connect(gain);
    gain.connect(this.master);
    source.start(start);
  }
}

export const sound = new SoundManager();
