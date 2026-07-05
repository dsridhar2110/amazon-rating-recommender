import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "./components/Nav";

export const metadata: Metadata = {
  title: "StarSense — Amazon Rating Recommender",
  description:
    "Guess how a shopper would rate a product (1–5 stars) — even a product no one has ever rated — and see, in plain words, why.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-canvas text-ink">
        <Nav />
        <main className="mx-auto max-w-6xl px-5 pb-24 pt-7">{children}</main>
        <footer className="mx-auto max-w-6xl px-5 py-8 text-xs leading-relaxed text-slate-500">
          StarSense — an independent portfolio project (originally the FIT5212 recommender-systems assignment).
          Data is a remix of real Amazon product-review ratings. The model is a hybrid of collaborative filtering,
          confidence-weighted statistics, and text similarity for never-seen items — a simple, honest baseline, not a
          production engine. Every prediction is explained.
        </footer>
      </body>
    </html>
  );
}
