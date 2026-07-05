import { Card } from "../components/ui";

const MAP = [
  ["Shopper", "Service engineer / customer site", "Who the prediction is personalised for"],
  ["Product (movie)", "Spare part · error code · fix · KB article", "The long catalogue, mostly rarely seen"],
  ["Star rating 1–5", "Predicted relevance / priority / satisfaction", "The score the model outputs"],
  ["Product name (text)", "Machine-log line · notification text", "Unstructured text → the “reads the name” (NLP) method"],
  ["Never-seen product", "A rare part / brand-new error signature", "The hard tail — exactly what cold-start protects"],
];

export default function AboutPage() {
  return (
    <div className="space-y-6">
      <section className="fade-up">
        <p className="text-xs font-semibold uppercase tracking-widest text-teal-dark">Service lens</p>
        <h1 className="mt-1 text-2xl font-extrabold tracking-tight text-ink">Why a movie recommender fits a service role</h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-slate-600">
          The engine is trained on <span className="font-semibold text-ink">real Amazon data</span> — that’s the honest,
          defensible core. This page is an <span className="font-semibold text-ink">analogy</span>, clearly labelled and
          never fabricated healthcare data: it shows how the <span className="font-semibold text-ink">same methods</span>{" "}
          transfer to a customer-service setting like Siemens Healthineers.
        </p>
      </section>

      <Card title="Same machine, different words" subtitle="Swap the nouns and the recommender becomes a service tool.">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
                <th className="pb-2 font-semibold">In this app</th>
                <th className="pb-2 font-semibold">At a service org</th>
                <th className="pb-2 font-semibold">Why it maps</th>
              </tr>
            </thead>
            <tbody>
              {MAP.map(([a, b, c]) => (
                <tr key={a} className="border-t border-line">
                  <td className="py-2.5 pr-4 font-semibold text-teal-dark">{a}</td>
                  <td className="py-2.5 pr-4 text-ink">{b}</td>
                  <td className="py-2.5 text-slate-500">{c}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card title="For the technical panel">
          <ul className="space-y-2 text-sm text-slate-600">
            <li>• A hybrid of collaborative filtering, statistics, and text similarity, with an explicit router.</li>
            <li>• Two honest test splits — a leakage-aware way to measure it.</li>
            <li>• RMSE over accuracy because ratings are skewed (most are 5★).</li>
            <li>• Served behind an API; monitored on a live page.</li>
          </ul>
        </Card>
        <Card title="For the business">
          <ul className="space-y-2 text-sm text-slate-600">
            <li>• Answers <span className="font-semibold text-ink">every</span> request, including brand-new items.</li>
            <li>• Each score comes with a plain reason → non-technical teams can trust it.</li>
            <li>• Quantifies the value of customer history (the 0.27 gap).</li>
          </ul>
        </Card>
        <Card title="For a CXO">
          <ul className="space-y-2 text-sm text-slate-600">
            <li>• Turns a <span className="font-semibold text-ink">long tail of rare cases</span> into ranked, prioritised actions.</li>
            <li>• Ships as a clickable, monitored product — not a notebook.</li>
            <li>• The same pattern re-skins to parts, logs and satisfaction with domain data.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
