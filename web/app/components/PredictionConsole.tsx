"use client";
import { useEffect, useState } from "react";
import {
  predict, getUsers, checkHealth, getDemoPredictions,
  PredictResult, TIER_META, API_BASE,
} from "../lib/api";
import { Card, RatingGauge, TierBadge } from "./ui";

const EXAMPLES: { label: string; user: number; name: string; note: string }[] = [
  { label: "A popular film", user: 1813, name: "A Clockwork Orange", note: "well-rated → “people like you”" },
  { label: "A different shopper", user: 534, name: "Almost Famous", note: "different taste" },
  { label: "Cold-start (made-up title)", user: 534, name: "The Completely Invented Space Western Vol 7", note: "never seen → reads the name" },
  { label: "Cold-start #2", user: 1944, name: "Midnight Robot Detective: The Neon Case", note: "cold-start reasoning" },
];

export function PredictionConsole() {
  const [userId, setUserId] = useState<number>(1813);
  const [name, setName] = useState<string>("A Clockwork Orange");
  const [users, setUsers] = useState<number[]>([]);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<"checking" | "live" | "offline">("checking");
  const [demo, setDemo] = useState<Record<string, PredictResult>>({});

  useEffect(() => {
    checkHealth().then((up) => {
      setMode(up ? "live" : "offline");
      if (up) getUsers(40).then(setUsers).catch(() => setUsers([]));
      else getDemoPredictions().then(setDemo);
    });
  }, []);

  async function run(u = userId, n = name) {
    setLoading(true); setErr(null);
    if (mode !== "live") {
      const hit = demo[`${u}::${n}`];
      if (hit) { setResult(hit); setLoading(false); return; }
      setErr(
        "This is the shared online demo, so only the example buttons are pre-computed. " +
          "Running it on my laptop (./run_local.sh) scores any shopper and any title you type."
      );
      setResult(null); setLoading(false);
      return;
    }
    try {
      setResult(await predict(u, n));
    } catch {
      setErr(`Couldn't reach the model at ${API_BASE}. Start it with: uvicorn ml.api:app --port 8000`);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      {/* Input side */}
      <Card title="Pick a shopper & a product" subtitle="Try a real film, or make up a title to see the cold-start trick.">
        <div className="mb-4 flex items-center gap-2 text-xs">
          {mode === "checking" && <span className="text-slate-500">· connecting…</span>}
          {mode === "live" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-teal/40 bg-teal-soft px-2.5 py-1 font-medium text-teal-dark">
              <span className="h-1.5 w-1.5 rounded-full bg-teal" /> live model — scores anything you type
            </span>
          )}
          {mode === "offline" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-2.5 py-1 font-medium text-amber-700">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" /> online demo — use the example buttons
            </span>
          )}
        </div>

        <label className="mb-1 block text-xs font-medium text-slate-600">
          Shopper <span className="font-normal text-slate-400">— a real reviewer, pick any number</span>
        </label>
        <input
          list="users" type="number" value={userId}
          onChange={(e) => setUserId(Number(e.target.value))}
          className="mb-4 w-full rounded-lg border border-line bg-white px-3 py-2 font-mono text-sm text-ink outline-none focus:border-teal"
        />
        <datalist id="users">{users.map((u) => <option key={u} value={u} />)}</datalist>

        <label className="mb-1 block text-xs font-medium text-slate-600">
          Product name <span className="font-normal text-slate-400">— a real title, or invent one to test cold-start</span>
        </label>
        <textarea
          value={name} rows={2} onChange={(e) => setName(e.target.value)}
          className="mb-3 w-full resize-none rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink outline-none focus:border-teal"
        />

        <div className="mb-4 flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              onClick={() => { setUserId(ex.user); setName(ex.name); run(ex.user, ex.name); }}
              className="rounded-full border border-line bg-white px-2.5 py-1 text-xs text-slate-600 transition hover:border-teal hover:text-teal-dark"
              title={ex.note}
            >
              {ex.label}
            </button>
          ))}
        </div>

        <button
          onClick={() => run()} disabled={loading}
          className="w-full rounded-lg bg-teal py-2.5 font-semibold text-white transition hover:bg-teal-dark disabled:opacity-50"
        >
          {loading ? "Predicting…" : "Predict the rating"}
        </button>
        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          <span className="font-semibold text-slate-600">Try this:</span> type a film that has never existed. The app
          still gives a sensible guess — that’s the cold-start method reading the title.
        </p>
      </Card>

      {/* Result side */}
      <Card title="The prediction, explained" subtitle="Every guess shows which of the three methods it used, and why.">
        {err && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800">{err}</div>
        )}
        {!err && !result && (
          <div className="grid h-56 place-items-center text-sm text-slate-400">
            Pick an example or hit “Predict the rating”.
          </div>
        )}
        {result && (
          <div className="fade-up">
            <div className="flex flex-col items-center">
              <RatingGauge value={result.rating} />
              <div className="mt-2"><TierBadge tier={result.tier} /></div>
            </div>

            <div className="mt-4 rounded-xl border border-line bg-slate-50 p-3.5 text-sm leading-relaxed text-slate-700">
              <span className="font-semibold text-ink">Why: </span>{result.reason}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <Fact k="Shopper" v={`#${result.user_id} · ${result.user_known ? "we know their history" : "new shopper"}`} />
              <Fact k="Method used" v={TIER_META[result.tier].short} />
            </div>

            {result.similar_items.length > 0 && (
              <div className="mt-4">
                <div className="mb-2 text-xs font-semibold text-slate-600">
                  It had never seen this title — so it borrowed from these similar-named products:
                </div>
                <ul className="space-y-1.5">
                  {result.similar_items.map((s) => (
                    <li key={s.name} className="flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-1.5 text-xs">
                      <span className="flex-1 truncate text-slate-700" title={s.name}>{s.name}</span>
                      <span className="font-mono text-content">match {s.similarity.toFixed(2)}</span>
                      <span className="font-mono text-slate-500">★ {s.mean_rating.toFixed(1)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <div className="text-slate-500">{k}</div>
      <div className="mt-0.5 font-semibold text-ink">{v}</div>
    </div>
  );
}
