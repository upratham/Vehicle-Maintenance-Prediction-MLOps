import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle2, DollarSign, Clock, Wrench } from "lucide-react";

export interface Impact {
  feature: string;
  value: string;
  contribution: number;
}

export interface ServiceEstimate {
  name: string;
  cost_low: number;
  cost_high: number;
  hours_low: number;
  hours_high: number;
  notes: string;
}

interface Props {
  result: string | null;
  score: number | null;
  service: ServiceEstimate | null;
  impacts: Impact[];
  error: string | null;
}

const fmtMoney = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);

export function ResultCard({ result, score, service, impacts, error }: Props) {
  const warn = result?.toLowerCase().includes("required");

  if (!result && !error) return null;

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

  const maxMag = Math.max(0.0001, ...impacts.map((i) => Math.abs(i.contribution)));

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={result ?? "r"}
        initial={{ opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.55, ease: [0.2, 0.8, 0.2, 1] }}
        className={`glass rounded-2xl relative overflow-hidden ${
          warn ? "border-ember-400/50" : "border-signal-green/40"
        }`}
        style={{ borderWidth: 1, borderStyle: "solid" }}
      >
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.9, delay: 0.1 }}
          style={{ transformOrigin: "left" }}
          className={`absolute top-0 left-0 right-0 h-[2px] ${warn ? "bg-ember-400" : "bg-signal-green"}`}
        />

        {/* Verdict */}
        <div className="p-8 md:p-10 flex items-start gap-5 border-b border-white/5">
          <div className="shrink-0">
            {warn ? (
              <AlertTriangle className="h-10 w-10 text-ember-400" />
            ) : (
              <CheckCircle2 className="h-10 w-10 text-signal-green" />
            )}
          </div>
          <div className="flex-1">
            <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
              Prediction Result
              {score !== null && (
                <span className="ml-3 font-mono text-bone-600 normal-case tracking-normal">
                  confidence {Math.round((warn ? score : 1 - score) * 100)}%
                </span>
              )}
            </div>
            <div className="font-display text-3xl md:text-4xl mt-2 text-bone-50 leading-tight">
              {result}
            </div>
            <div className="text-bone-400 text-sm mt-3 max-w-md">
              {warn
                ? "Schedule a service inspection soon. Estimated service details below."
                : "No immediate maintenance indicated. Continue routine monitoring."}
            </div>
          </div>
        </div>

        {/* Service estimate */}
        {warn && service && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, duration: 0.5 }}
            className="p-8 md:p-10 grid grid-cols-1 md:grid-cols-3 gap-6 border-b border-white/5"
          >
            <div>
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.28em] text-bone-400 mb-2">
                <Wrench className="h-3.5 w-3.5" /> Likely Service
              </div>
              <div className="font-display text-xl text-bone-50">{service.name}</div>
              <div className="text-bone-600 text-xs mt-1">{service.notes}</div>
            </div>
            <div>
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.28em] text-bone-400 mb-2">
                <DollarSign className="h-3.5 w-3.5" /> Estimated Cost
              </div>
              <div className="font-display text-2xl text-ember-300">
                {fmtMoney(service.cost_low)} <span className="text-bone-400 text-lg">–</span> {fmtMoney(service.cost_high)}
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.28em] text-bone-400 mb-2">
                <Clock className="h-3.5 w-3.5" /> Shop Time
              </div>
              <div className="font-display text-2xl text-bone-50">
                {service.hours_low}
                <span className="text-bone-400 text-lg"> – </span>
                {service.hours_high}
                <span className="text-bone-400 text-base font-body font-normal ml-1">hrs</span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Feature impact */}
        {impacts.length > 0 && (
          <div className="p-8 md:p-10">
            <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400 mb-5">
              Why · Feature Impact Analysis
            </div>
            <div className="space-y-2.5">
              {impacts.slice(0, 6).map((i, idx) => {
                const pct = (Math.abs(i.contribution) / maxMag) * 100;
                const pos = i.contribution >= 0;
                return (
                  <motion.div
                    key={i.feature}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.35 + idx * 0.06 }}
                    className="grid grid-cols-[1fr_2fr_auto] items-center gap-4 text-sm"
                  >
                    <div>
                      <div className="text-bone-100">{i.feature}</div>
                      <div className="text-bone-600 text-xs font-mono">{i.value}</div>
                    </div>
                    <div className="relative h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ delay: 0.4 + idx * 0.06, duration: 0.6, ease: [0.2, 0.8, 0.2, 1] }}
                        className={`absolute top-0 bottom-0 left-0 rounded-full ${
                          pos ? "bg-ember-400" : "bg-signal-green/80"
                        }`}
                      />
                    </div>
                    <div
                      className={`font-mono text-xs tabular-nums ${pos ? "text-ember-300" : "text-signal-green"}`}
                    >
                      {pos ? "+" : ""}
                      {(i.contribution * 100).toFixed(0)}%
                    </div>
                  </motion.div>
                );
              })}
            </div>
            <div className="mt-5 text-[11px] text-bone-600 font-mono">
              Positive bars push toward "maintenance required"; negative bars push against it.
            </div>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
