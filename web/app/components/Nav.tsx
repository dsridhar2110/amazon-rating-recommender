"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Try it" },
  { href: "/monitor", label: "How it's doing" },
  { href: "/about", label: "Service lens" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-white/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-teal text-base font-bold text-white">★</span>
          <span className="leading-tight">
            <span className="block font-semibold tracking-tight text-ink">StarSense</span>
            <span className="-mt-0.5 block text-[11px] text-slate-500">Amazon Rating Recommender</span>
          </span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {links.map((l) => {
            const active = path === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-lg px-3 py-1.5 font-medium transition ${
                  active ? "bg-teal-soft text-teal-dark" : "text-slate-500 hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
