// Typed client for the MalariaScope FastAPI backend.

const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type Label = "Parasitized" | "Uninfected";

export interface Prediction {
  filename: string;
  label: Label;
  probability_parasitized: number;
  confidence: number;
  threshold: number;
  needs_review: boolean;
  warnings: string[];
  gradcam_png: string | null;
  latency_ms: number;
}

export interface BatchSummary {
  total: number;
  analyzed: number;
  failed: number;
  parasitized: number;
  uninfected: number;
  needs_review: number;
  flagged_input: number;
  estimated_parasitemia_pct: number;
  latency_ms: number;
}

export interface BatchResponse {
  results: Prediction[];
  errors: { filename: string; error: string }[];
  summary: BatchSummary;
}

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model: string | null;
  run: string | null;
  uptime_s: number;
  detail: string | null;
}

export interface BinaryMetrics {
  threshold: number;
  accuracy: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  npv: number;
  f1: number;
  balanced_accuracy: number;
  mcc: number;
  confusion: { tp: number; fp: number; tn: number; fn: number };
}

export interface ReliabilityBin {
  lo: number;
  hi: number;
  confidence: number;
  frequency: number;
  count: number;
}

export interface SplitStats {
  cells: number;
  parasitized: number;
  uninfected: number;
  patients: number;
}

export interface Metrics {
  model: string;
  split_mode: "patient" | "cell";
  params: { total?: number; trainable?: number };
  img_size: number;
  train_seconds: number | null;
  eval_ms_per_image_batched: number;
  splits: { train: SplitStats; val: SplitStats; test: SplitStats; leakage: Record<string, number> };
  calibration: {
    temperature: number;
    ece_before: number;
    ece_after: number;
    brier_before: number;
    brier_after: number;
    reliability_before: ReliabilityBin[];
    reliability_after: ReliabilityBin[];
  };
  threshold: {
    policy: string;
    target_sensitivity: number;
    value: number;
    youden_value: number;
    validation: BinaryMetrics;
    review_margin: number;
  };
  validation: { roc_auc: number; pr_auc: number; n: number };
  test: {
    roc_auc: number;
    pr_auc: number;
    n: number;
    at_threshold: BinaryMetrics;
    at_0_5: BinaryMetrics;
    ci95_patient_bootstrap: Partial<Record<"accuracy" | "sensitivity" | "specificity" | "f1" | "roc_auc", [number, number]>>;
    review_rate: number;
    accuracy_excluding_review: number | null;
    roc_curve: { fpr: number; tpr: number }[];
    pr_curve: { recall: number; precision: number }[];
  };
  history: Record<string, number[]>;
  finetune_start_epoch: number | null;
}

export interface ModelInfo {
  metadata: {
    model: string;
    run: string;
    trained_at: string;
    img_size: number;
    temperature: number;
    threshold: number;
    review_margin: number;
    target_sensitivity: number;
    split_mode: string;
    dataset: string;
  };
  metrics: Metrics | null;
  disclaimer: string;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${BASE}/api/health`).then((r) => handle<Health>(r)),
  model: () => fetch(`${BASE}/api/model`).then((r) => handle<ModelInfo>(r)),
  predictBatch: (files: File[], gradcam: boolean, signal?: AbortSignal) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    return fetch(`${BASE}/api/predict/batch?gradcam=${gradcam}`, { method: "POST", body: form, signal }).then((r) =>
      handle<BatchResponse>(r),
    );
  },
};

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
