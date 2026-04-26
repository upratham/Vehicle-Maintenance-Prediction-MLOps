import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Moon, Sun } from "lucide-react";
import { VinPanel } from "./components/VinPanel";
import { FallbackPanel } from "./components/FallbackPanel";
import { ConditionPanel, type ConditionValues } from "./components/ConditionPanel";
import { ResultCard, type Impact, type ServiceEstimate } from "./components/ResultCard";
import { HyundaiPanel, type HyundaiInputs } from "./components/HyundaiPanel";
import { EnginePanel, type EngineInputs } from "./components/EnginePanel";
import { AnomalyResultCard } from "./components/AnomalyResultCard";
import { Ops } from "./pages/Ops";
import type { AutoFilled } from "./lib/mapping";
import { HYUNDAI_RANGES, ENGINE_RANGES, interpret, type Interpretation } from "./lib/anomalyRanges";

type Mode = "vin" | "manual";
type Theme = "dark" | "light";
type View = "predict" | "ops";
type Model = "maintenance" | "hyundai" | "engine";

const MODEL_TABS: { id: Model; label: string; subtitle: string }[] = [
  { id: "maintenance", label: "Vehicle Maintenance", subtitle: "Year/make/model · service forecast" },
  { id: "hyundai", label: "Hyundai Anomaly", subtitle: "Sensor stream · anomaly detection" },
  { id: "engine", label: "Engine Condition", subtitle: "6-sensor engine diagnosis" },
];

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

  const [model, setModel] = useState<Model>("maintenance");

  // Maintenance model state
  const [mode, setMode] = useState<Mode>("vin");
  const [auto, setAuto] = useState<AutoFilled | null>(null);
  const [vin, setVin] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [service, setService] = useState<ServiceEstimate | null>(null);
  const [impacts, setImpacts] = useState<Impact[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Anomaly model state (shared by hyundai + engine)
  const [anomalyStatus, setAnomalyStatus] = useState<string | null>(null);
  const [anomalyScore, setAnomalyScore] = useState<number | null>(null);
  const [anomalyInterp, setAnomalyInterp] = useState<Interpretation | null>(null);
  const [anomalyError, setAnomalyError] = useState<string | null>(null);

  const resetAnomaly = () => {
    setAnomalyStatus(null);
    setAnomalyScore(null);
    setAnomalyInterp(null);
    setAnomalyError(null);
  };

  const switchModel = (m: Model) => {
    setModel(m);
    resetAnomaly();
    setResult(null);
    setError(null);
    setScore(null);
    setService(null);
    setImpacts([]);
  };

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

  const runHyundai = async (v: HyundaiInputs) => {
    setPredicting(true);
    resetAnomaly();
    try {
      const r = await fetch("/predict/hyundai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(v),
      });
      if (!r.ok) throw new Error(`Prediction failed (${r.status})`);
      const j = await r.json();
      const status = j.status ?? j.label ?? "No result";
      setAnomalyStatus(status);
      setAnomalyScore(typeof j.score === "number" ? j.score : null);
      setAnomalyInterp(
        interpret(
          {
            engine_temperature: v.engine_temperature,
            brake_pad_thickness: v.brake_pad_thickness,
            tire_pressure: v.tire_pressure,
          },
          HYUNDAI_RANGES,
          status
        )
      );
    } catch (e: unknown) {
      setAnomalyError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  const runEngine = async (v: EngineInputs) => {
    setPredicting(true);
    resetAnomaly();
    try {
      const r = await fetch("/predict/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(v),
      });
      if (!r.ok) throw new Error(`Prediction failed (${r.status})`);
      const j = await r.json();
      const status = j.status ?? j.label ?? "No result";
      setAnomalyStatus(status);
      setAnomalyScore(typeof j.score === "number" ? j.score : null);
      setAnomalyInterp(interpret(v as unknown as Record<string, number>, ENGINE_RANGES, status));
    } catch (e: unknown) {
      setAnomalyError(e instanceof Error ? e.message : "Prediction failed");
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
          className="flex items-center justify-between mb-12"
        >
          <div className="flex items-center gap-3">
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

        {view === "ops" && <Ops />}

        {view === "predict" && (
          <>
            {/* Model picker */}
            <div className="mb-10 flex flex-wrap gap-2 border-b border-white/5 pb-4">
              {MODEL_TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => switchModel(t.id)}
                  className={`px-4 py-2 text-[11px] uppercase tracking-[0.22em] font-mono rounded-md border transition-colors ${
                    model === t.id
                      ? "bg-ember-400 border-ember-400 text-ink-900"
                      : "border-white/10 text-bone-400 hover:text-ember-300 hover:border-ember-400/40"
                  }`}
                  title={t.subtitle}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {model === "maintenance" && (
              <>
                <motion.section
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.6 }}
                  className="text-center mb-10"
                >
                  <h1 className="font-display font-light text-[36px] md:text-[52px] leading-[1.05] tracking-tight text-bone-50">
                    Read a vehicle's <em className="text-ember-400 not-italic">pulse</em>
                  </h1>
                  <p className="mt-4 mx-auto max-w-md text-bone-400 text-sm md:text-base leading-relaxed">
                    Scan a VIN or pick a year, make &amp; model to forecast upcoming service needs.
                  </p>
                </motion.section>

                <div className="space-y-6">
                  {!auto && (
                    <div className="flex justify-end">
                      <div className="inline-flex items-center rounded-full border border-white/10 p-0.5 font-mono text-[10px] uppercase tracking-[0.22em]">
                        <button
                          onClick={() => setMode("vin")}
                          className={`px-3.5 py-1.5 rounded-full transition-colors ${
                            mode === "vin" ? "bg-ember-400 text-ink-900" : "text-bone-400 hover:text-ember-300"
                          }`}
                        >
                          VIN
                        </button>
                        <button
                          onClick={() => setMode("manual")}
                          className={`px-3.5 py-1.5 rounded-full transition-colors ${
                            mode === "manual" ? "bg-ember-400 text-ink-900" : "text-bone-400 hover:text-ember-300"
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
                          <VinPanel onDecoded={(a, v) => { setAuto(a); setVin(v); }} />
                        </motion.div>
                      )}
                      {!auto && mode === "manual" && (
                        <motion.div key="manual" exit={{ opacity: 0, y: -10 }}>
                          <FallbackPanel onSubmit={(a) => { setAuto(a); setVin(null); }} />
                        </motion.div>
                      )}
                      {auto && (
                        <motion.div key="cond">
                          <ConditionPanel auto={auto} vin={vin} onPredict={runPredict} predicting={predicting} />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  <ResultCard result={result} score={score} service={service} impacts={impacts} error={error} />

                  {auto && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-center">
                      <button
                        onClick={reset}
                        className="text-[11px] uppercase tracking-[0.28em] text-bone-600 hover:text-ember-300 transition-colors"
                      >
                        ↺  Start over with another vehicle
                      </button>
                    </motion.div>
                  )}
                </div>
              </>
            )}

            {model === "hyundai" && (
              <div className="space-y-6">
                <SectionIntro
                  title="Hyundai sensor stream"
                  subtitle="Anomaly detection model trained on Hyundai maintenance telemetry."
                />
                <HyundaiPanel onPredict={runHyundai} predicting={predicting} />
                <AnomalyResultCard
                  status={anomalyStatus}
                  score={anomalyScore}
                  interpretation={anomalyInterp}
                  error={anomalyError}
                  resultLabel="Anomaly Detection Result"
                />
              </div>
            )}

            {model === "engine" && (
              <div className="space-y-6">
                <SectionIntro
                  title="Engine diagnosis"
                  subtitle="Six engine sensors classified against healthy operating bands."
                />
                <EnginePanel onPredict={runEngine} predicting={predicting} />
                <AnomalyResultCard
                  status={anomalyStatus}
                  score={anomalyScore}
                  interpretation={anomalyInterp}
                  error={anomalyError}
                  resultLabel="Engine Condition Result"
                />
              </div>
            )}
          </>
        )}

      </div>
    </div>
  );
}

function SectionIntro({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="text-center mb-2"
    >
      <h1 className="font-display font-light text-[28px] md:text-[36px] leading-[1.1] tracking-tight text-bone-50">
        {title}
      </h1>
      <p className="mt-3 mx-auto max-w-md text-bone-400 text-sm leading-relaxed">{subtitle}</p>
    </motion.section>
  );
}
