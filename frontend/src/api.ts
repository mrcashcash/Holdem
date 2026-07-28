import type {
  CashReloadRequest,
  ChampionQueryRequest,
  ChampionQueryResult,
  ChampionSpotRequest,
  ChampionSpotState,
  GameSettings,
  GameState,
  LiveScreenDecisionFeed,
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

async function requestLiveScreenDecision(): Promise<LiveScreenDecisionFeed> {
  const response = await fetch('http://127.0.0.1:8765/latest', {
    cache: 'no-store',
  })
  if (!response.ok) {
    throw new Error('The live screen watcher is not available.')
  }
  return response.json() as Promise<LiveScreenDecisionFeed>
}

function subscribeLiveScreenDecision(
  onDecision: (decision: LiveScreenDecisionFeed) => void,
  onError: () => void,
): () => void {
  const events = new EventSource('http://127.0.0.1:8765/events')
  events.onmessage = (event) => {
    try {
      onDecision(JSON.parse(event.data) as LiveScreenDecisionFeed)
    } catch {
      onError()
    }
  }
  events.onerror = onError
  return () => events.close()
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
  agentAction: () => request<GameState>('/game/agent-action', { method: 'POST' }),
  queryChampion: (query: ChampionQueryRequest) => request<ChampionQueryResult>('/champion/query', {
    method: 'POST', body: JSON.stringify(query),
  }),
  previewChampionSpot: (spot: ChampionSpotRequest) => request<ChampionSpotState>('/champion/spot', {
    method: 'POST', body: JSON.stringify(spot),
  }),
  liveScreenDecision: requestLiveScreenDecision,
  subscribeLiveScreenDecision,
  trainingStatus: () => request<TrainingStatus>('/training/status'),
  train: (episodes: number) => request<TrainingStatus>('/training/start', {
    method: 'POST', body: JSON.stringify({ episodes }),
  }),
  reloadLastModel: () => request<ModelLoadResult>('/training/reload-last', {
    method: 'POST',
  }),
}
