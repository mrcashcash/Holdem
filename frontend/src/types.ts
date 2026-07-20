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
}

export interface TrainingArtifacts {
  abstraction: boolean
  blueprint: boolean
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
  serving_agent: 'GpuBlueprintAgent' | 'BlueprintAgent' | 'HeuristicAgent'
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
