export interface BaselineModel {
  name: string;
  family: string;
  f1: number;
  precision: number;
  recall: number;
  roc_auc: number;
  accuracy: number;
  winner?: boolean;
  hyperparams?: Record<string, unknown>;
}

export interface ModelInfo {
  profile_name?: string;
  version?: string;
  trained_at?: string;
  sha256?: string;
  s3_uri?: string;
  model_family?: string;
  framework?: string;
  promoted?: boolean;
  metrics?: {
    f1?: number;
    precision?: number;
    recall?: number;
    roc_auc?: number;
    accuracy?: number;
  };
  params?: Record<string, unknown>;
  baselines?: BaselineModel[];
  baselines_computed_at?: string;
}

export type AllModelInfo = Record<string, ModelInfo>;

export const PROFILE_LABELS: Record<string, string> = {
  vehicle_maintenance: "Vehicle Maintenance",
  cars_hyundai: "Vehicle Anomaly",
  engine_data: "Engine Condition",
};

export interface DriftFeature {
  feature: string;
  type: "numeric" | "categorical";
  psi: number;
  verdict: "stable" | "warning" | "drifted";
  n_live: number;
}

export interface DriftReport {
  window: string;
  source: "mongo" | "demo" | "none";
  n_predictions: number;
  thresholds: { stable: number; warning: number };
  features: DriftFeature[];
  error?: string;
}

export async function fetchModelInfo(profile?: string): Promise<ModelInfo> {
  const url = profile ? `/model_info?profile=${encodeURIComponent(profile)}` : "/model_info";
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model_info failed (${r.status})`);
  return r.json();
}

export async function fetchAllModelInfo(): Promise<AllModelInfo> {
  const r = await fetch("/model_info/all");
  if (!r.ok) throw new Error(`model_info/all failed (${r.status})`);
  return r.json();
}

export async function fetchDrift(window = "7d", profile = "vehicle_maintenance"): Promise<DriftReport> {
  const r = await fetch(`/drift?window=${encodeURIComponent(window)}&profile=${encodeURIComponent(profile)}`);
  if (!r.ok) throw new Error(`drift failed (${r.status})`);
  return r.json();
}

export type RetrainEvent =
  | { kind: "line"; line: string }
  | { kind: "summary"; payload: RetrainSummary }
  | { kind: "done" }
  | { kind: "error"; message: string };

export interface RetrainSummary {
  event: "done" | "error";
  promoted?: boolean;
  old_metrics?: ModelInfo["metrics"];
  new_metrics?: ModelInfo["metrics"];
  new_version?: string;
  error?: string;
}

/**
 * EventSource doesn't support custom headers, so we POST via fetch with a
 * ReadableStream and parse the SSE frames ourselves.
 */
export async function startRetrain(
  opsToken: string,
  onEvent: (e: RetrainEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/train", {
    method: "POST",
    headers: { "X-Ops-Token": opsToken },
    signal,
  });

  if (!res.ok || !res.body) {
    const msg = res.status === 401 ? "invalid ops token" : `retrain failed (${res.status})`;
    onEvent({ kind: "error", message: msg });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";

    for (const frame of frames) {
      const lines = frame.split("\n");
      let event = "message";
      let data = "";
      for (const l of lines) {
        if (l.startsWith("event:")) event = l.slice(6).trim();
        else if (l.startsWith("data:")) data += l.slice(5).trim();
      }
      if (!data && event !== "done") continue;

      if (event === "summary") {
        try {
          onEvent({ kind: "summary", payload: JSON.parse(data) });
        } catch {
          /* ignore parse errors */
        }
      } else if (event === "done") {
        onEvent({ kind: "done" });
      } else {
        try {
          const obj = JSON.parse(data);
          if (obj.line) onEvent({ kind: "line", line: obj.line });
        } catch {
          /* heartbeat comments, ignore */
        }
      }
    }
  }

  onEvent({ kind: "done" });
}
