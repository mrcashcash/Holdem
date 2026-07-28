export interface LegalActions {
  fold?: boolean
  check?: boolean
  call?: boolean
  raise?: boolean
  all_in?: boolean
  to_call?: number
  current_bet?: number
  player_bet?: number
  raise_min?: number
  raise_max?: number
}

export interface PlayerSessionStats {
  hand_wins: number
  match_wins: number
  total_buy_in?: number
  showdown_wins: number
  fold_wins: number
  folds: number
  calls: number
  raises: number
  vpip: number
  pfr: number
  aggression: number | null
}

export interface SessionStats {
  hands_completed: number
  split_pots: number
  showdown_hands: number
  total_pot: number
  biggest_pot: number
  players: PlayerSessionStats[]
}

export interface GameSettings {
  initial_stack: number
  small_blind: number
  big_blind: number
}

export interface CashReloadRequest {
  player: 0 | 1 | 'both'
  amount: number
}

export interface GameState {
  hand_number: number
  street: string
  button: number
  current_player: number | null
  stacks: number[]
  pot: number
  last_pot: number
  round_bets: number[]
  to_call: number
  hero_cards: string[]
  hero_hand_strength: string | null
  opponent_cards: string[]
  community: string[]
  history: string[]
  legal_actions: LegalActions
  complete: boolean
  result: string | null
  winner: number | null
  session_stats: SessionStats
  settings?: GameSettings
}

export interface TrainingArtifacts {
  abstraction: boolean
  blueprint: boolean
}

export interface ServingDepth {
  depth_bb: number
  iteration: number
}

export interface ServingModelStatus {
  kind: 'MultiStackBlueprintAgent' | 'GpuBlueprintAgent' | 'BlueprintAgent' | 'HeuristicAgent'
  selected_depth_bb: number | null
  iteration: number | null
  available_depths: ServingDepth[]
  search_enabled: boolean
  search_iterations: number | null
}

export interface TrainingStatus {
  running: boolean
  episodes: number
  completed: number
  progress: number
  last_error: string | null
  updates: number
  parameters: number
  iterations_per_second: number
  serving_agent: 'MultiStackBlueprintAgent' | 'GpuBlueprintAgent' | 'BlueprintAgent' | 'HeuristicAgent'
  serving_model?: ServingModelStatus
  river_search: boolean
  artifacts: TrainingArtifacts
  trainer: string
}

export interface ChampionHistoryAction {
  player: 0 | 1
  action: 'fold' | 'check' | 'call' | 'raise' | 'all_in'
  amount?: number
}

export interface ChampionSpotRequest {
  hero_cards?: string[]
  board?: string[]
  button?: 0 | 1
  stacks?: number[]
  actions?: ChampionHistoryAction[]
}

export interface ChampionQueryRequest extends ChampionSpotRequest {
  current?: boolean
}

export interface ChampionSpotMetrics {
  effective_stack: number
  effective_stack_bb: number
  spr: number
  pot_odds_percent: number
  hand_strength: string | null
}

export interface ChampionSpotState {
  hero_cards: string[]
  board: string[]
  staged_board: string[]
  button: 0 | 1
  street: string
  current_player: 0 | 1 | null
  starting_stacks: number[]
  stacks: number[]
  round_bets: number[]
  pot: number
  to_call: number
  legal_actions: LegalActions
  complete: boolean
  result: string | null
  required_board_count: number | null
  metrics: ChampionSpotMetrics
}

export interface ChampionQueryAction {
  action: string
  label: string
  amount: number | null
  probability: number
  percentage: number
}

export interface ChampionQueryResult {
  source: string
  iteration: number
  street: string
  position: string
  pot: number
  to_call: number
  legal_actions: LegalActions
  exact_match: boolean
  node: number | null
  bucket: number | null
  actions: ChampionQueryAction[]
  recommended: ChampionQueryAction
  warnings: string[]
  spot: ChampionSpotState | null
}

export type LiveScreenDecisionStatus =
  | 'waiting'
  | 'thinking'
  | 'ready'
  | 'stale'
  | 'error'

export interface LiveScreenTableState {
  captured_at: string
  hand_number: number | null
  street: string | null
  pot: number | null
  stacks: Array<number | null>
  round_bets: Array<number | null>
  hero_cards: string[]
  board: string[]
  button: 0 | 1 | null
  current_player: 0 | 1 | null
  complete: boolean
  stable: boolean
  history_stable: boolean
  confidence: number
  recognition_ms: number | null
  warnings: string[]
  players: string[]
  visible_actions: LiveScreenHistoryAction[]
  timeline_starts_at_hand: boolean
}

export interface LiveScreenHistoryAction {
  player: 0 | 1
  action: string
  amount: number | null
  street: string
}

export type LiveScreenHistoryStatus =
  | 'in_progress'
  | 'verified'
  | 'verified_actions'
  | 'partial'
  | 'gap'

export interface LiveScreenHistoryStep {
  id: string
  captured_at: string
  street: string | null
  pot: number | null
  stacks: Array<number | null>
  board: string[]
  current_player: 0 | 1 | null
  complete: boolean
  confidence: number
  recognition_ms: number | null
  transition: string
  verified: boolean
  recovered: boolean
  warnings: string[]
  actions: LiveScreenHistoryAction[]
  decision: LiveScreenDecision | null
}

export interface LiveScreenHandHistory {
  id: string
  hand_number: number | null
  started_at: string
  updated_at: string
  status: LiveScreenHistoryStatus
  verification_message: string
  hero_cards: string[]
  players: string[]
  button: 0 | 1 | null
  complete: boolean
  recovered: boolean
  steps: LiveScreenHistoryStep[]
  decisions: LiveScreenDecision[]
}

export interface LiveScreenStrategyAction {
  action: string
  label?: string
  amount: number | null
  server_amount?: number | null
  probability: number
  percentage?: number
}

export interface LiveScreenDecision {
  decision_id: string
  hand_number: number | null
  captured_at: string
  decided_at: string
  action: string
  amount: number | null
  all_in: boolean
  model: string
  iteration: number | null
  recognition_confidence: number
  street: string
  pot: number
  to_call: number
  hero_cards: string[]
  board: string[]
  stacks: [number, number]
  warnings: string[]
  strategy: LiveScreenStrategyAction[]
  latency_ms: number | null
  total_latency_ms: number | null
  source: string | null
}

export interface LiveScreenDecisionFeed {
  connected: boolean
  status: LiveScreenDecisionStatus
  message: string
  updated_at: string
  amount_scale: number
  table: LiveScreenTableState | null
  decision: LiveScreenDecision | null
  history: LiveScreenHandHistory[]
  history_gap_count: number
}
