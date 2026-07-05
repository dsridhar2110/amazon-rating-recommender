import type { Config } from "tailwindcss";

// Light theme — matches the interview-walkthrough.html (white background, teal accent).
const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f8fafc",   // page background
        paper: "#ffffff",    // cards
        ink: "#0f172a",      // primary text
        line: "#e2e8f0",     // borders
        teal: { DEFAULT: "#0d9488", dark: "#0f766e", soft: "#f0fdfa" },
        // the three model strategies (readable on white)
        svd: "#0d9488",
        stats: "#7c3aed",
        content: "#d97706",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
