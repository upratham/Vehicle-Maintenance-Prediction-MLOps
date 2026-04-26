import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";
import type { Interpretation, Verdict } from "../lib/anomalyRanges";

interface Props {
  status: string | null;
  score: number | null;
  interpretation: Interpretation | null;
  error: string | null;
  resultLabel: string;
}

const verdictRing: Record<Verdict, string> = {
  ok: "bg-signal-green/15 text-signal-green border-signal-green/30",
  watch: "bg-ember-400/15 text-ember-300 border-ember-400/40",
  alert: "bg-signal-red/15 text-signal-red border-signal-red/40",
};

const verdictWord: Record<Verdict, string> = {
  ok: "OK",
  watch: "Watch",
  alert: "Alert",
};

export function AnomalyResultCard({ status, score, interpretation, error, resultLabel }: Props) {
  if (!status && !error) return null;

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-8 border border-signal-red/40"
      >
        <div className="flex items-center gap-4">
          <AlertTriangle className="h-8 w-8 text-signal-red" />
          <div>
            <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">System Error</div>
            <div className="font-display text-2xl mt-1 text-bone-50">{error}</div>
          </div>
        </div>
      </motion.div>
    );
  }

  const anomaly =
    typeof status === "string" && /anomaly|fault|warning|required|attention/i.test(status);
  const Icon = anomaly ? ShieldAlert : CheckCircle2;
  const accent = anomaly ? "border-ember-400/50" : "border-signal-green/40";
  const stripe = anomaly ? "bg-ember-400" : "bg-signal-green";
  const iconClr = anomaly ? "text-ember-400" : "text-signal-green";

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={status ?? "r"}
        initial={{ opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.55, ease: [0.2, 0.8, 0.2, 1] }}
        className={`glass rounded-2xl relative overflow-hidden ${accent}`}
        style={{ borderWidth: 1, borderStyle: "solid" }}
      >
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.9, delay: 0.1 }}
          style={{ transformOrigin: "left" }}
          className={`absolute top-0 left-0 right-0 h-[2px] ${stripe}`}
        />

        <div className="p-8 md:p-10 flex items-start gap-5 border-b border-white/5">
          <Icon className={`h-10 w-10 shrink-0 ${iconClr}`} />
          <div className="flex-1">
            <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
              {resultLabel}
              {score !== null && (
                <span className="ml-3 font-mono text-bone-600 normal-case tracking-normal">
                  score {score.toFixed(3)}
                </span>
              )}
            </div>
            <div className="font-display text-3xl md:text-4xl mt-2 text-bone-50 leading-tight">
              {interpretation?.headline ?? status}
            </div>
            {interpretation?.summary && (
              <div className="text-bone-300 text-sm mt-3 leading-relaxed">{interpretation.summary}</div>
            )}
            <div className="text-bone-600 text-[11px] font-mono mt-3 uppercase tracking-[0.22em]">
              model verdict · {status}
            </div>
          </div>
        </div>

        {interpretation && (
          <div className="p-8 md:p-10 space-y-3">
            <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400 mb-2">
              Per-feature reading
            </div>
            {interpretation.features.map((f, idx) => (
              <motion.div
                key={f.range.key}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.25 + idx * 0.06 }}
                className="grid grid-cols-[auto_1fr] items-start gap-4"
              >
                <span
                  className={`text-[10px] font-mono uppercase tracking-[0.18em] px-2 py-1 rounded border ${verdictRing[f.verdict]}`}
                >
                  {verdictWord[f.verdict]}
                </span>
                <div className="text-sm text-bone-200 leading-relaxed">{f.sentence}</div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
