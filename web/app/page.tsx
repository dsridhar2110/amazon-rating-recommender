import { PredictionConsole } from "./components/PredictionConsole";
import { Card } from "./components/ui";

const SAMPLE = [
  { u: "#1813", p: "Beautiful Thing", s: 5 },
  { u: "#1944", p: "Almost Famous", s: 5 },
  { u: "#534", p: "A Clockwork Orange", s: 5 },
];

const TIERS = [
  { c: "#0d9488", n: "“People like you”", d: "When a product has enough ratings, it learns taste patterns — people who agreed with you before will probably agree again. (The technical name is collaborative filtering / SVD.) Handles ~77% of cases." },
  { c: "#7c3aed", n: "Sensible averages", d: "When a product has only been rated a few times, it blends this shopper's habits with the product's average — leaning on whichever has more data behind it." },
  { c: "#d97706", n: "Reads the name", d: "When a product has never been rated, it reads the product's name (that's the NLP part), finds the closest-named known products, and borrows their ratings. This is the cold-start case." },
];

export default function Home() {
  return (
    <div className="space-y-7">
      {/* Intro */}
      <section className="fade-up">
        <p className="text-xs font-semibold uppercase tracking-widest text-teal-dark">Amazon Rating Recommender</p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight text-ink">
          Guess how a shopper would rate a product — before they’ve tried it.
        </h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-slate-600">
          This app looks at how thousands of people rated products in the past and uses it to predict the
          <span className="font-semibold text-ink"> star rating (1–5)</span> a shopper would give something new — even a
          product <span className="font-semibold text-ink">no one has ever rated</span>. And for every guess, it tells you
          in plain words <span className="font-semibold text-ink">which method it used and why.</span>
        </p>
      </section>

      {/* Data snapshot */}
      <Card title="What the data looks like" subtitle="One row = one shopper rating one product. Every prediction comes from data like this.">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
                <th className="pb-2 font-semibold">Shopper</th>
                <th className="pb-2 font-semibold">Product (name)</th>
                <th className="pb-2 text-right font-semibold">Stars given</th>
              </tr>
            </thead>
            <tbody>
              {SAMPLE.map((r) => (
                <tr key={r.u} className="border-t border-line">
                  <td className="py-2 text-slate-600">{r.u}</td>
                  <td className="py-2 text-ink">{r.p}</td>
                  <td className="py-2 text-right font-mono font-semibold text-teal-dark">{"★".repeat(r.s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-[13px] leading-relaxed text-slate-600">
          <span className="font-semibold text-ink">The scale:</span> ~745,000 ratings · 2,000 shoppers · 201,000
          products. <span className="font-semibold text-ink">The quirk that drives everything:</span> the typical product
          has been rated just <span className="font-semibold text-ink">once</span>, and about{" "}
          <span className="font-semibold text-ink">9%</span> of products have never been rated at all — which is exactly
          why the app has a special way to handle brand-new items.
        </p>
      </Card>

      {/* The interactive console */}
      <section>
        <h2 className="text-lg font-bold tracking-tight text-ink">Try a prediction</h2>
        <p className="mt-1 text-sm text-slate-600">
          Pick a shopper, pick or type a product, and see the predicted rating — with its explanation.
        </p>
        <div className="mt-3">
          <PredictionConsole />
        </div>
      </section>

      {/* How it decides */}
      <Card title="How it decides — three methods, one smart switch" subtitle="It picks the right method for how much it knows about the product, and always tells you which one it used.">
        <div className="grid gap-3 sm:grid-cols-3">
          {TIERS.map((t) => (
            <div key={t.n} className="rounded-xl border border-line bg-slate-50 p-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: t.c }} />
                <span className="text-sm font-semibold text-ink">{t.n}</span>
              </div>
              <p className="text-[13px] leading-relaxed text-slate-600">{t.d}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
