export interface HealthResponse {
  app_status: string;
  package_version: string;
  api_status: string;
  timestamp: string;
}

export interface SystemStatusResponse {
  app_version: string;
  output_dir: string;
  run_index_path: string;
  profiles_count: number;
  datasets_count: number;
  model_providers: string[];
  models_count: number;
  warnings: string[];
  timestamp: string;
}

export interface ProfileSummary {
  profile_id: string;
  name: string;
  description?: string | null;
  path: string;
  evaluators: string[];
  model_names: string[];
  available: boolean;
  valid: boolean;
  validation_message?: string | null;
}

export interface ProfileDetail {
  profile_id: string;
  name: string;
  description?: string | null;
  path: string;
  run_name?: string | null;
  output_dir?: string | null;
  dataset?: string | null;
  evaluators: Array<Record<string, unknown>>;
  model_names: string[];
  valid: boolean;
  validation_message?: string | null;
  available?: boolean;
}

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  path: string;
  format?: string | null;
  sample_count?: number | null;
  valid: boolean;
  validation_message?: string | null;
}

export interface ModelSummary {
  provider: string;
  name: string;
  available: boolean;
  connection_status: string;
  warning_message?: string | null;
  source_profile?: string | null;
}

export interface RunSummary {
  run_id: string;
  run_name: string;
  status: string;
  dry_run: boolean;
  output_dir?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  created_at?: string | null;
  dataset_path?: string | null;
  model_names: string[];
  evaluator_names: string[];
  error_message?: string | null;
  in_progress?: boolean;
  ready?: boolean;
  message?: string | null;
}

export interface RunDetail extends RunSummary {
  config_hash?: string | null;
  artifact_paths?: Record<string, unknown> | null;
}

export interface CreateRunResponse {
  run_id: string;
  status: string;
  message?: string | null;
  created_at: string;
  output_dir: string;
  dry_run: boolean;
}

export interface EvaluationRecord {
  metric_name: string;
  score: number;
  passed: boolean;
  details?: Record<string, unknown>;
}

export interface DetailedSample {
  sample_idx: number;
  prompt: string;
  expected: string;
  model_response: string;
  latency_ms?: number;
  evaluations: EvaluationRecord[];
}

export interface ModelResult {
  model_name?: string;
  summary?: Array<Record<string, string>>;
  detailed?: DetailedSample[];
}

export interface RunResultsResponse {
  run_id: string;
  run_name?: string | null;
  status?: string | null;
  output_dir?: string | null;
  model_results: ModelResult[];
  comparison?: Record<string, unknown> | null;
  ready: boolean;
  message?: string | null;
}

export interface RunAuditResponse {
  run_id: string;
  status?: string | null;
  audit?: Record<string, unknown> | null;
  ready: boolean;
  message?: string | null;
}

export interface ArtifactFile {
  kind: string;
  path: string;
  exists: boolean;
  model_name?: string | null;
}

export interface RunArtifactsResponse {
  run_id: string;
  output_dir?: string | null;
  status?: string | null;
  files: ArtifactFile[];
  ready: boolean;
  message?: string | null;
}

export interface ReviewItem {
  sample_idx: number;
  prompt: string;
  expected: string;
  model_response: string;
  score: number;
  passed: boolean;
  metric_name: string;
}

export interface NotableMetric {
  label: string;
  value: string;
  context: string;
}

export interface ExecutiveSummary {
  run_id: string;
  generated_at: string;
  evaluation_purpose: string;
  overall_outcome: string;
  recommended_next_step: string;
  key_strengths: string[];
  key_weaknesses: string[];
  needs_human_review_count: number;
  notable_metrics: NotableMetric[];
  notes?: string | null;
}

export interface ExecutiveSummaryResponse {
  run_id: string;
  status?: string | null;
  ready: boolean;
  message?: string | null;
  summary?: ExecutiveSummary | null;
}
