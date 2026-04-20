/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0a0a0b",
          800: "#111114",
          700: "#1a1a1f",
          600: "#24242b",
          500: "#3a3a44",
        },
        bone: {
          50:  "#f5f2ec",
          100: "#ece7dc",
          200: "#d8d1c2",
          400: "#9a9384",
          600: "#6b6659",
        },
        ember: {
          300: "#ffcf7a",
          400: "#f5a524",
          500: "#e0890b",
          600: "#b3660a",
        },
        signal: {
          green: "#7fb069",
          red:   "#c94b4b",
        },
      },
      fontFamily: {
        display: ['"Fraunces"', 'ui-serif', 'Georgia', 'serif'],
        body:    ['"Geist"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono:    ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        inset1: "inset 0 1px 0 0 rgba(255,255,255,0.04)",
        glow:   "0 0 0 1px rgba(245,165,36,0.25), 0 20px 60px -20px rgba(245,165,36,0.35)",
      },
    },
  },
  plugins: [],
};
