"use client";
import { useEffect, useState } from "react";
import { getMetrics, Metrics, TIER_META, Tier } from "../lib/api";
import { Card, Stat, BarRow, Donut, Explainer } from "./ui";

const TIER_ORDER: Tier[] = ["ENHANCED_SVD", "HYBRID_STATISTICS", "CONTENT_SIMILARITY"];

export function MonitorDashboard() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getMetrics().then(setM).catch(() => setError(true));
  }, []);

  if (error) return <p className="text-sm text-amber-700">Could not load the results.</p>;
  if (!m) return <p className="text-sm text-slate-500">Loading results…</p>;

  const baselineRows = Object.entries(m.baselines);
  const maxRmse = Math.max(m.hybrid.rmse, ...baselineRows.map(([, b]) => b.rmse)) * 1.05;
  const bestBaseline = Math.min(...baselineRows.map(([, b]) => b.rmse));
  const lift = (((bestBaseline - m.hybrid.rmse) / bestBaseline) * 100).toFixed(1);

  const donut = TIER_ORDER.map((t) => ({
    label: TIER_META[t].short, value: m.tier_mix_pct[t] ?? 0, color: TIER_META[t].color,
  }));

  return (
    <div className="space-y-5">
      {/* header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-teal-dark">How it’s doing</p>
          <h1 className="mt-1 text-2xl font-extrabold tracking-tight text-ink">Is the model any good?</h1>
          <p className="mt-1 max-w-2xl text-sm text-slate-600">
            These numbers are measured on ratings the model never saw during training — a fair test, not a boast.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <Pill>{m.dataset.total_ratings.toLocaleString()} ratings</Pill>
          <Pill>{m.dataset.users.toLocaleString()} shoppers</Pill>
          <Pill>{m.dataset.products.toLocaleString()} products</Pill>
        </div>
      </div>

      <Explainer title="What have we done here?">
        We built a system that guesses a shopper’s star rating for a product, then tested it by hiding some real
        ratings, predicting them, and checking how close it got. Everything below is that check — how big the typical
        mistake is, which of the three methods did the work, and whether the scores can be trusted at face value.
      </Explainer>

      {/* headline tiles */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Typical error (RMSE)" value={m.hybrid.rmse.toFixed(3)} sub={`${lift}% better than the best lazy guess`} accent="#0d9488" />
        <Stat label="Average miss (MAE)" value={m.hybrid.mae.toFixed(3)} sub="stars, on average" />
        <Stat label="Fit (R²)" value={m.hybrid.r2.toFixed(3)} sub="higher = explains more" />
        <Stat label="Coverage" value={`${m.coverage_pct.toFixed(0)}%`} sub="answers every request" accent="#7c3aed" />
      </div>
      <Explainer title="What do these mean?">
        <b>RMSE</b> is the typical size of the mistake, in stars — lower is better; {m.hybrid.rmse.toFixed(2)} means a
        guess is usually within about {m.hybrid.rmse.toFixed(1)} of a star. <b>MAE</b> is the plain average miss.
        <b> R²</b> is how much of the ups-and-downs the model explains (0 = no better than guessing the average).
        <b> Coverage</b> = it never leaves a request unanswered, including brand-new products.
      </Explainer>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* baselines */}
        <Card title="Does it beat the lazy guesses?" subtitle="Typical error (RMSE) — shorter bars are better.">
          <div className="space-y-2.5">
            <BarRow label="Our model" value={m.hybrid.rmse} max={maxRmse} color="#0d9488" highlight />
            {baselineRows.map(([name, b]) => (
              <BarRow key={name} label={pretty(name)} value={b.rmse} max={maxRmse} color="#cbd5e1" />
            ))}
          </div>
          <div className="mt-3">
            <Explainer title="What’s a “baseline”?">
              The lazy versions to beat — “just guess the overall average,” “guess this shopper’s average,” “guess this
              product’s average.” If the real model can’t beat those, it hasn’t earned its keep. Ours beats all three.
            </Explainer>
          </div>
        </Card>

        {/* tier mix */}
        <Card title="Which method did the work?" subtitle="Share of predictions handled by each of the three methods.">
          <Donut data={donut} />
          <div className="mt-3">
            <Explainer title="The three methods, in one line each">
              <b>“People like you”</b> (collaborative) learns taste patterns from everyone’s ratings.
              <b> Sensible averages</b> (statistical) blends the shopper’s and product’s averages when data is thin.
              <b> Reads the name</b> (cold-start / NLP) turns a never-seen product’s title into numbers and borrows from
              similar-named products.
            </Explainer>
          </div>
        </Card>
      </div>

      {/* per-tier */}
      <Card title="Where it’s strong, where it’s hard" subtitle="Error rises as the model knows less about a product — the cold-start case is the honest hard part.">
        <div className="grid gap-3 sm:grid-cols-3">
          {TIER_ORDER.map((t) => {
            const pt = m.per_tier_rmse[t];
            if (!pt) return null;
            return (
              <div key={t} className="rounded-xl border border-line bg-slate-50 p-4">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: TIER_META[t].color }} />
                  <span className="text-sm font-semibold text-ink">{TIER_META[t].label}</span>
                </div>
                <div className="mt-3 font-mono text-2xl font-semibold" style={{ color: TIER_META[t].color }}>
                  {pt.rmse.toFixed(3)}
                </div>
                <div className="text-xs text-slate-500">typical error · {pt.n.toLocaleString()} guesses · {m.tier_mix_pct[t]}% of traffic</div>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* calibration */}
        <Card title="Can you trust the score at face value?" subtitle="Calibration — when it says 4.5, is the truth really about 4.5?">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500">
                <th className="pb-2 font-semibold">When it predicts…</th>
                <th className="pb-2 text-right font-semibold">count</th>
                <th className="pb-2 text-right font-semibold">avg predicted</th>
                <th className="pb-2 text-right font-semibold">avg actual</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {m.calibration.map((c) => (
                <tr key={c.bucket} className="border-t border-line">
                  <td className="py-1.5 text-slate-500">{c.bucket}</td>
                  <td className="py-1.5 text-right text-slate-400">{c.n.toLocaleString()}</td>
                  <td className="py-1.5 text-right text-teal-dark">{c.mean_pred.toFixed(2)}</td>
                  <td className="py-1.5 text-right text-ink">{c.mean_actual.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3">
            <Explainer title="What is calibration?">
              We group the guesses into bands and compare the <b>average predicted</b> with the <b>average actual</b> in
              each band. If those two columns stay close, a “4.5” really means about 4.5 — so a person (or a business)
              can act on the number without second-guessing it.
            </Explainer>
          </div>
        </Card>

        {/* distribution + eval-design note */}
        <div className="space-y-5">
          <Card title="Do the guesses look like real ratings?" subtitle="The spread of all predictions — a sanity check.">
            <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
              <Mini k="average" v={m.prediction_distribution.mean.toFixed(2)} />
              <Mini k="spread" v={m.prediction_distribution.std.toFixed(2)} />
              <Mini k="range" v={`${m.prediction_distribution.min.toFixed(1)}–${m.prediction_distribution.max.toFixed(1)}`} />
              <Mini k="under 3★" v={`${m.prediction_distribution.pct_below_3}%`} />
              <Mini k="over 4.5★" v={`${m.prediction_distribution.pct_above_4_5}%`} />
              <Mini k="never-seen items" v={`${m.val_cold_items_pct}%`} />
            </div>
            <div className="mt-3">
              <Explainer title="What is the prediction distribution?">
                Real Amazon ratings lean high (lots of 5s), so the guesses should too. If they bunched up somewhere
                strange, that would warn us something’s off. This is a quick reality check, not a score.
              </Explainer>
            </div>
          </Card>
          <Card title="How we tested it fairly">
            <p className="text-[13px] leading-relaxed text-slate-600">
              We hid some ratings and kept every shopper known (that’s how it would really run) — {m.val_cold_users_pct}%
              new shoppers, {m.val_cold_items_pct}% never-seen products. A deliberately harsh test with{" "}
              <b className="text-ink">all-new shoppers</b> scores about <b className="text-ink">1.08</b>. The gap to our{" "}
              <b className="text-ink">{m.hybrid.rmse.toFixed(2)}</b> is a real finding: it’s how much knowing a shopper’s
              history is worth.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}

function pretty(name: string) {
  return { global_mean: "Guess the average", user_mean: "Guess shopper’s avg", item_mean: "Guess product’s avg" }[name] || name;
}
function Pill({ children }: { children: React.ReactNode }) {
  return <span className="rounded-full border border-line bg-white px-2.5 py-1 text-slate-500">{children}</span>;
}
function Mini({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <div className="text-[11px] text-slate-500">{k}</div>
      <div className="mt-0.5 font-mono font-semibold text-ink">{v}</div>
    </div>
  );
}
