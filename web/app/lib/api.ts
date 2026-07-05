// Client for the RecoPulse FastAPI model service.
// Base URL is configurable so the same build runs against a local uvicorn
// (default) or a hosted model service in production.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export type Tier = "ENHANCED_SVD" | "HYBRID_STATISTICS" | "CONTENT_SIMILARITY";

export interface SimilarItem {
  name: string;
  similarity: number;
  mean_rating: number;
}

export interface PredictResult {
  rating: number;
  tier: Tier;
  reason: string;
  similar_items: SimilarItem[];
  user_id: number;
  product_name: string;
  user_known: boolean;
}

export interface Metrics {
  split: string;
  validation_rows: number;
  val_cold_users_pct: number;
  val_cold_items_pct: number;
  hybrid: { rmse: number; mae: number; r2: number };
  baselines: Record<string, { rmse: number; mae: number; r2: number }>;
  tier_mix_pct: Record<Tier, number>;
  tier_mix_counts: Record<Tier, number>;
  per_tier_rmse: Record<Tier, { n: number; rmse: number; mae: number; r2: number }>;
  coverage_pct: number;
  calibration: { bucket: string; n: number; mean_pred: number; mean_actual: number }[];
  prediction_distribution: {
    mean: number; std: number; min: number; max: number;
    pct_below_3: number; pct_above_4_5: number;
  };
  dataset: { total_ratings: number; users: number; products: number };
  hyperparams: Record<string, unknown>;
}

export async function predict(user_id: number, product_name: string): Promise<PredictResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, product_name }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

// Is a live model API reachable? Short timeout so the UI decides quickly.
export async function checkHealth(): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal, cache: "no-store" });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

// Bundled snapshot of precomputed predictions, so the preset demos (incl. the
// cold-start moment) work on the public Vercel URL with no backend running.
export async function getDemoPredictions(): Promise<Record<string, PredictResult>> {
  try {
    const res = await fetch("/demo_predictions.json", { cache: "force-cache" });
    return res.ok ? res.json() : {};
  } catch {
    return {};
  }
}

export async function getUsers(limit = 40): Promise<number[]> {
  const res = await fetch(`${API_BASE}/users?limit=${limit}`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return (await res.json()).user_ids;
}

// Metrics: try the live API, fall back to the bundled static snapshot so the
// monitoring cockpit renders even with no backend running (e.g. on Vercel).
export async function getMetrics(): Promise<Metrics> {
  try {
    const res = await fetch(`${API_BASE}/metrics`, { cache: "no-store" });
    if (res.ok) return res.json();
  } catch {
    /* fall through to static */
  }
  const res = await fetch("/metrics.json", { cache: "no-store" });
  return res.json();
}

export const TIER_META: Record<Tier, { label: string; color: string; short: string }> = {
  ENHANCED_SVD: { label: "“People like you” (collaborative)", color: "#0d9488", short: "People like you" },
  HYBRID_STATISTICS: { label: "Sensible averages (statistical)", color: "#7c3aed", short: "Averages" },
  CONTENT_SIMILARITY: { label: "Reads the name (cold-start · NLP)", color: "#d97706", short: "Cold-start" },
};
