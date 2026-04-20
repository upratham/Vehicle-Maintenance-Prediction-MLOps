import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Moon, Sun } from "lucide-react";
import { VinPanel } from "./components/VinPanel";
import { FallbackPanel } from "./components/FallbackPanel";
import { ConditionPanel, type ConditionValues } from "./components/ConditionPanel";
import { ResultCard, type Impact, type ServiceEstimate } from "./components/ResultCard";
import { Ops } from "./pages/Ops";
import type { AutoFilled } from "./lib/mapping";

type Mode = "vin" | "manual";
type Theme = "dark" | "light";
type View = "predict" | "ops";

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("theme") as Theme | null;
    return saved ?? "dark";
  });

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("light", theme === "light");
    localStorage.setItem("theme", theme);
  }, [theme]);

  const [view, setView] = useState<View>(() =>
    (typeof window !== "undefined" && window.location.hash === "#ops") ? "ops" : "predict"
  );

  useEffect(() => {
    const onHash = () => setView(window.location.hash === "#ops" ? "ops" : "predict");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const [mode, setMode] = useState<Mode>("vin");
  const [auto, setAuto] = useState<AutoFilled | null>(null);
  const [vin, setVin] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [service, setService] = useState<ServiceEstimate | null>(null);
  const [impacts, setImpacts] = useState<Impact[]>([]);
  const [error, setError] = useState<string | null>(null);

  const runPredict = async (
    payload: ConditionValues & { auto: AutoFilled; vin: string | null }
  ) => {
    setPredicting(true);
    setError(null);
    setResult(null);
    setService(null);
    setImpacts([]);
    setScore(null);
    try {
      const body = {
        Vehicle_Age: payload.auto.Vehicle_Age ?? 0,
        Engine_Size: payload.auto.Engine_Size ?? 0,
        Odometer_Reading: payload.Odometer_Reading,
        Fuel_Efficiency: payload.Fuel_Efficiency,
        Reported_Issues: payload.Reported_Issues,
        Accident_History: payload.Accident_History,
        Tire_Condition: payload.Tire_Condition,
        Brake_Condition: payload.Brake_Condition,
        Battery_Status: payload.Battery_Status,
        Vehicle_Model: payload.auto.Vehicle_Model ?? "Car",
        Fuel_Type: payload.auto.Fuel_Type ?? "Petrol",
        Transmission_Type: payload.auto.Transmission_Type ?? "Automatic",
      };
      const r = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`Prediction failed (${r.status})`);
      const j = await r.json();
      setResult(j.status ?? "No result");
      setScore(typeof j.score === "number" ? j.score : null);
      setService(j.service ?? null);
      setImpacts(Array.isArray(j.impacts) ? j.impacts : []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  const reset = () => {
    setAuto(null);
    setVin(null);
    setResult(null);
    setScore(null);
    setService(null);
    setImpacts([]);
    setError(null);
  };

  return (
    <div className="grain relative min-h-screen">
      <div className="fixed inset-0 -z-10 ember-glow" />
      <div className="fixed inset-0 -z-10 grid-bg opacity-40" />

      <div className="max-w-3xl mx-auto px-6 md:px-10 pt-10 pb-24">
        <motion.header
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="flex items-center justify-between mb-16"
        >
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-full border border-ember-400/50 grid place-items-center">
              <div className="h-2 w-2 bg-ember-400 rounded-full animate-pulse" />
            </div>
            <span className="font-mono text-xs uppercase tracking-[0.3em] text-bone-400">
              Maintenance&nbsp;/&nbsp;Predictor
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="inline-flex items-center rounded-full border border-white/10 p-0.5 font-mono text-[10px] uppercase tracking-[0.22em]">
              <button
                onClick={() => { window.location.hash = ""; setView("predict"); }}
                className={`px-3 py-1 rounded-full transition-colors ${
                  view === "predict" ? "bg-ember-400 text-ink-900" : "text-bone-400 hover:text-ember-300"
                }`}
              >
                Predict
              </button>
              <button
                onClick={() => { window.location.hash = "ops"; setView("ops"); }}
                className={`px-3 py-1 rounded-full transition-colors ${
                  view === "ops" ? "bg-ember-400 text-ink-900" : "text-bone-400 hover:text-ember-300"
                }`}
              >
                Ops
              </button>
            </div>
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle theme"
              className="h-8 w-8 rounded-full border border-white/10 hover:border-ember-400/60 grid place-items-center text-bone-400 hover:text-ember-300 transition-colors"
            >
              {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            </button>
          </div>
        </motion.header>

        {view === "predict" && (
          <motion.section
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.9, delay: 0.15 }}
            className="text-center mb-10"
          >
            <h1 className="font-display font-light text-[36px] md:text-[52px] leading-[1.05] tracking-tight text-bone-50">
              Read a vehicle's <em className="text-ember-400 not-italic">pulse</em>
            </h1>
            <p className="mt-4 mx-auto max-w-md text-bone-400 text-sm md:text-base leading-relaxed">
              Scan a VIN or pick a year, make &amp; model to forecast upcoming service needs.
            </p>
          </motion.section>
        )}

        {view === "ops" && <Ops />}

        {view === "predict" && <div className="space-y-6">
          {!auto && (
            <div className="flex justify-end">
              <div className="inline-flex items-center rounded-full border border-white/10 p-0.5 font-mono text-[10px] uppercase tracking-[0.22em]">
                <button
                  onClick={() => setMode("vin")}
                  className={`px-3.5 py-1.5 rounded-full transition-colors ${
                    mode === "vin"
                      ? "bg-ember-400 text-ink-900"
                      : "text-bone-400 hover:text-ember-300"
                  }`}
                >
                  VIN
                </button>
                <button
                  onClick={() => setMode("manual")}
                  className={`px-3.5 py-1.5 rounded-full transition-colors ${
                    mode === "manual"
                      ? "bg-ember-400 text-ink-900"
                      : "text-bone-400 hover:text-ember-300"
                  }`}
                >
                  Year / Make / Model
                </button>
              </div>
            </div>
          )}

          <div className={!auto ? "min-h-[380px]" : undefined}>
            <AnimatePresence mode="wait">
              {!auto && mode === "vin" && (
                <motion.div key="vin" exit={{ opacity: 0, y: -10 }}>
                  <VinPanel
                    onDecoded={(a, v) => {
                      setAuto(a);
                      setVin(v);
                    }}
                  />
                </motion.div>
              )}
              {!auto && mode === "manual" && (
                <motion.div key="manual" exit={{ opacity: 0, y: -10 }}>
                  <FallbackPanel
                    onSubmit={(a) => {
                      setAuto(a);
                      setVin(null);
                    }}
                  />
                </motion.div>
              )}
              {auto && (
                <motion.div key="cond">
                  <ConditionPanel
                    auto={auto}
                    vin={vin}
                    onPredict={runPredict}
                    predicting={predicting}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <ResultCard result={result} score={score} service={service} impacts={impacts} error={error} />

          {auto && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-center"
            >
              <button
                onClick={reset}
                className="text-[11px] uppercase tracking-[0.28em] text-bone-600 hover:text-ember-300 transition-colors"
              >
                ↺  Start over with another vehicle
              </button>
            </motion.div>
          )}
        </div>}

        <footer className="mt-24 pt-8 border-t border-white/5 flex items-center justify-center text-[11px] font-mono text-bone-600">
          <span>MSML605 · MLOps · v0.1</span>
        </footer>
      </div>
    </div>
  );
}
