import { TIER_META, Tier } from "../lib/api";

export function Card({
  title, subtitle, children, className = "",
}: { title?: string; subtitle?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-line bg-paper p-5 shadow-sm ${className}`}>
      {title && (
        <div className="mb-4">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label, value, sub, accent = "#0f172a",
}: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-line bg-slate-50 p-4">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-2xl font-semibold" style={{ color: accent }}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function TierBadge({ tier }: { tier: Tier }) {
  const m = TIER_META[tier];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold"
      style={{ background: `${m.color}18`, color: m.color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: m.color }} />
      {m.label}
    </span>
  );
}

// A 1–5 rating gauge (semi-circular arc).
export function RatingGauge({ value }: { value: number }) {
  const pct = (value - 1) / 4; // 0..1
  const R = 80, CX = 100, CY = 100;
  const start = Math.PI, end = 0;
  const ang = start + (end - start) * pct;
  const x = CX + R * Math.cos(ang), y = CY - R * Math.sin(ang);
  const arc = (a0: number, a1: number) =>
    `M ${CX + R * Math.cos(a0)} ${CY - R * Math.sin(a0)} A ${R} ${R} 0 0 1 ${CX + R * Math.cos(a1)} ${CY - R * Math.sin(a1)}`;
  const color = value >= 4 ? "#0d9488" : value >= 3 ? "#7c3aed" : "#d97706";
  return (
    <svg viewBox="0 0 200 120" className="w-full max-w-[280px]">
      <path d={arc(start, end)} stroke="#e2e8f0" strokeWidth="14" fill="none" strokeLinecap="round" />
      <path d={arc(start, ang)} stroke={color} strokeWidth="14" fill="none" strokeLinecap="round" />
      <circle cx={x} cy={y} r="8" fill={color} />
      <text x="100" y="86" textAnchor="middle" fill="#0f172a" className="font-mono" fontSize="34" fontWeight="700">
        {value.toFixed(2)}
      </text>
      <text x="100" y="104" textAnchor="middle" fill="#64748b" fontSize="10">
        predicted stars (1–5)
      </text>
    </svg>
  );
}

// Horizontal comparison bars (e.g., our model vs simple baselines — lower is better).
export function BarRow({
  label, value, max, color, highlight = false, unit = "",
}: { label: string; value: number; max: number; color: string; highlight?: boolean; unit?: string }) {
  const pct = Math.max(6, (value / max) * 100);
  return (
    <div className="flex items-center gap-3 text-sm">
      <div className={`w-32 shrink-0 ${highlight ? "font-semibold text-ink" : "text-slate-500"}`}>{label}</div>
      <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-slate-100">
        <div
          className="flex h-full items-center justify-end rounded-md pr-2 font-mono text-xs text-white"
          style={{ width: `${pct}%`, background: color }}
        >
          {value.toFixed(3)}{unit}
        </div>
      </div>
    </div>
  );
}

// Donut for the strategy mix.
export function Donut({ data }: { data: { label: string; value: number; color: string }[] }) {
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  const R = 60, C = 2 * Math.PI * R;
  let offset = 0;
  return (
    <div className="flex items-center gap-6">
      <svg viewBox="0 0 160 160" className="h-40 w-40 -rotate-90">
        {data.map((d) => {
          const frac = d.value / total;
          const dash = frac * C;
          const seg = (
            <circle
              key={d.label} cx="80" cy="80" r={R} fill="none" stroke={d.color} strokeWidth="20"
              strokeDasharray={`${dash} ${C - dash}`} strokeDashoffset={-offset}
            />
          );
          offset += dash;
          return seg;
        })}
        <circle cx="80" cy="80" r="42" fill="#ffffff" />
      </svg>
      <ul className="space-y-2 text-sm">
        {data.map((d) => (
          <li key={d.label} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: d.color }} />
            <span className="text-slate-600">{d.label}</span>
            <span className="ml-auto font-mono text-slate-500">{((d.value / total) * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// A small "what this means" explainer note — reused across the app.
export function Explainer({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-teal/20 bg-teal-soft p-3.5">
      <div className="text-xs font-semibold text-teal-dark">{title}</div>
      <p className="mt-1 text-[13px] leading-relaxed text-slate-600">{children}</p>
    </div>
  );
}
