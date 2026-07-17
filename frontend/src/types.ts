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
  serving_agent: 'BlueprintAgent' | 'HeuristicAgent'
  river_search: boolean
  artifacts: TrainingArtifacts
  trainer: string
}
