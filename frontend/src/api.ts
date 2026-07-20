import type {
  CashReloadRequest,
  ChampionQueryRequest,
  ChampionQueryResult,
  ChampionSpotRequest,
  ChampionSpotState,
  GameSettings,
  GameState,
  TrainingStatus,
} from './types'

export interface ModelLoadResult {
  ok: boolean
  agent: string
  source: string
  iteration: number | null
  status: TrainingStatus
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? 'The server rejected that request.')
  }
  return response.json() as Promise<T>
}

export const api = {
  getGame: () => request<GameState>('/game'),
  newGame: () => request<GameState>('/game/new', { method: 'POST' }),
  nextHand: () => request<GameState>('/game/next', { method: 'POST' }),
  updateGameSettings: (settings: GameSettings) => request<GameState>('/game/settings', {
    method: 'POST', body: JSON.stringify(settings),
  }),
  reloadCash: (reload: CashReloadRequest) => request<GameState>('/game/reload-cash', {
    method: 'POST', body: JSON.stringify(reload),
  }),
  action: (action: string, amount?: number) => request<GameState>('/game/action', {
    method: 'POST', body: JSON.stringify({ action, amount }),
  }),
  queryChampion: (query: ChampionQueryRequest) => request<ChampionQueryResult>('/champion/query', {
    method: 'POST', body: JSON.stringify(query),
  }),
  previewChampionSpot: (spot: ChampionSpotRequest) => request<ChampionSpotState>('/champion/spot', {
    method: 'POST', body: JSON.stringify(spot),
  }),
  trainingStatus: () => request<TrainingStatus>('/training/status'),
  train: (episodes: number) => request<TrainingStatus>('/training/start', {
    method: 'POST', body: JSON.stringify({ episodes }),
  }),
  reloadLastModel: () => request<ModelLoadResult>('/training/reload-last', {
    method: 'POST',
  }),
}
