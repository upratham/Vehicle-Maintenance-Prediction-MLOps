import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Cpu, Database, GitBranch, Play, Trophy } from "lucide-react";
import {
  fetchAllModelInfo,
  fetchDrift,
  PROFILE_LABELS,
  startRetrain,
  type AllModelInfo,
  type DriftReport,
  type ModelInfo,
  type RetrainSummary,
} from "../lib/ops";

const PROFILE_ORDER = ["vehicle_maintenance", "cars_hyundai", "engine_data"] as const;
type ProfileName = (typeof PROFILE_ORDER)[number];

const fmtPct = (n?: number) => (typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "—");

function MetricCell({ label, value }: { label: string; value?: number }) {
  return (
    <div>
      <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-bone-600">{label}</div>
      <div className="font-display text-lg text-bone-50 mt-0.5 tabular-nums">{fmtPct(value)}</div>
    </div>
  );
}

function RegistryCard({ info }: { info: ModelInfo | null }) {
  if (!info) {
    return (
      <div className="glass rounded-2xl p-8 border border-white/10">
        <div className="text-bone-400 font-mono text-xs">Loading model registry…</div>
      </div>
    );
  }
  const trained = info.trained_at ? new Date(info.trained_at).toLocaleString() : "—";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl border border-white/10 overflow-hidden"
    >
      <div className="p-6 md:p-8 border-b border-white/5 flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400 flex items-center gap-2">
            <GitBranch className="h-3.5 w-3.5" /> Model Registry
          </div>
          <div className="font-display text-2xl md:text-3xl text-bone-50 mt-2">
            {info.version ?? "—"}
          </div>
          <div className="text-bone-500 text-xs mt-1">{info.model_family}</div>
        </div>
        {info.promoted && (
          <span className="text-[10px] uppercase tracking-[0.22em] font-mono text-signal-green border border-signal-green/40 rounded-full px-2.5 py-1">
            Promoted
          </span>
        )}
      </div>
      <div className="p-6 md:p-8 grid grid-cols-2 md:grid-cols-5 gap-5">
        <MetricCell label="F1" value={info.metrics?.f1} />
        <MetricCell label="Precision" value={info.metrics?.precision} />
        <MetricCell label="Recall" value={info.metrics?.recall} />
        <MetricCell label="ROC-AUC" value={info.metrics?.roc_auc} />
        <MetricCell label="Accuracy" value={info.metrics?.accuracy} />
      </div>
      <div className="p-6 md:p-8 border-t border-white/5 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
        <div>
          <div className="text-bone-600 uppercase tracking-[0.22em] mb-1">Trained</div>
          <div className="text-bone-300">{trained}</div>
        </div>
        <div>
          <div className="text-bone-600 uppercase tracking-[0.22em] mb-1">S3</div>
          <div className="text-bone-300 truncate">{info.s3_uri ?? "—"}</div>
        </div>
        <div className="md:col-span-2">
          <div className="text-bone-600 uppercase tracking-[0.22em] mb-1">SHA-256</div>
          <div className="text-bone-400 break-all">{info.sha256 ?? "—"}</div>
        </div>
      </div>
    </motion.div>
  );
}

function BaselinesTable({ info }: { info: ModelInfo | null }) {
  const rows = info?.baselines ?? [];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.05 }}
      className="glass rounded-2xl border border-white/10 overflow-hidden"
    >
      <div className="p-6 md:p-8 border-b border-white/5 flex items-center gap-2">
        <Trophy className="h-4 w-4 text-ember-400" />
        <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
          Baseline Comparison
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-[10px] font-mono uppercase tracking-[0.22em] text-bone-600 border-b border-white/5">
              <th className="text-left px-6 py-3 font-normal">Model</th>
              <th className="text-right px-3 py-3 font-normal">F1</th>
              <th className="text-right px-3 py-3 font-normal">Precision</th>
              <th className="text-right px-3 py-3 font-normal">Recall</th>
              <th className="text-right px-3 py-3 font-normal">ROC-AUC</th>
              <th className="text-right px-6 py-3 font-normal">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr
                key={m.name}
                className={`border-b border-white/5 ${m.winner ? "bg-ember-400/5" : ""}`}
              >
                <td className="px-6 py-3">
                  <div className="flex items-center gap-2">
                    {m.winner && <Trophy className="h-3.5 w-3.5 text-ember-400" />}
                    <span className={m.winner ? "text-ember-300 font-medium" : "text-bone-100"}>
                      {m.name}
                    </span>
                  </div>
                  <div className="text-[10px] text-bone-600 font-mono mt-0.5">{m.family}</div>
                </td>
                <td className="px-3 py-3 text-right font-mono tabular-nums text-bone-100">{fmtPct(m.f1)}</td>
                <td className="px-3 py-3 text-right font-mono tabular-nums text-bone-300">{fmtPct(m.precision)}</td>
                <td className="px-3 py-3 text-right font-mono tabular-nums text-bone-300">{fmtPct(m.recall)}</td>
                <td className="px-3 py-3 text-right font-mono tabular-nums text-bone-300">{fmtPct(m.roc_auc)}</td>
                <td className="px-6 py-3 text-right font-mono tabular-nums text-bone-300">{fmtPct(m.accuracy)}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-bone-500 text-xs font-mono">
                  no baselines recorded yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}

function DriftCard({ report }: { report: DriftReport | null }) {
  if (!report) {
    return (
      <div className="glass rounded-2xl p-8 border border-white/10">
        <div className="text-bone-400 font-mono text-xs">Computing drift…</div>
      </div>
    );
  }
  const max = Math.max(0.5, ...report.features.map((f) => f.psi));
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="glass rounded-2xl border border-white/10 overflow-hidden"
    >
      <div className="p-6 md:p-8 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-ember-400" />
          <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
            Population Stability Index · window {report.window}
          </div>
        </div>
        <div className="text-[10px] font-mono text-bone-600">
          n={report.n_predictions} · {report.source}
        </div>
      </div>
      <div className="p-6 md:p-8 space-y-2.5">
        {report.features.map((f) => {
          const pct = Math.min(100, (f.psi / max) * 100);
          const color =
            f.verdict === "stable"
              ? "bg-signal-green/70"
              : f.verdict === "warning"
              ? "bg-ember-400"
              : "bg-signal-red";
          return (
            <div key={f.feature} className="grid grid-cols-[1fr_2fr_auto] gap-4 items-center text-sm">
              <div>
                <div className="text-bone-100">{f.feature}</div>
                <div className="text-bone-600 text-[10px] font-mono uppercase tracking-[0.22em]">
                  {f.verdict} · {f.type}
                </div>
              </div>
              <div className="relative h-1.5 bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.6 }}
                  className={`absolute inset-y-0 left-0 rounded-full ${color}`}
                />
              </div>
              <div className="font-mono text-xs tabular-nums text-bone-300">
                {f.psi.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="px-6 md:px-8 pb-6 text-[10px] font-mono text-bone-600">
        PSI &lt; 0.1 stable · 0.1–0.25 warning · &gt; 0.25 drifted
      </div>
    </motion.div>
  );
}

function MetricDiff({ label, before, after }: { label: string; before?: number; after?: number }) {
  const delta = typeof before === "number" && typeof after === "number" ? after - before : null;
  const sign = delta && delta > 0 ? "+" : "";
  const cls = delta == null ? "text-bone-500" : delta > 0 ? "text-signal-green" : delta < 0 ? "text-signal-red" : "text-bone-400";
  return (
    <div className="flex items-baseline justify-between py-1.5 border-b border-white/5">
      <div className="text-[11px] font-mono uppercase tracking-[0.22em] text-bone-600">{label}</div>
      <div className="text-xs font-mono tabular-nums">
        <span className="text-bone-400">{fmtPct(before)}</span>
        <span className="text-bone-600 mx-2">→</span>
        <span className="text-bone-100">{fmtPct(after)}</span>
        {delta !== null && (
          <span className={`ml-3 ${cls}`}>{sign}{(delta * 100).toFixed(2)}%</span>
        )}
      </div>
    </div>
  );
}

function RetrainPanel({ onSummary }: { onSummary: () => void }) {
  const [token, setToken] = useState<string>(() => sessionStorage.getItem("opsToken") ?? "");
  const [running, setRunning] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [summary, setSummary] = useState<RetrainSummary | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines]);

  const run = async () => {
    setRunning(true);
    setLines([]);
    setSummary(null);
    sessionStorage.setItem("opsToken", token);
    await startRetrain(token, (e) => {
      if (e.kind === "line") setLines((l) => [...l, e.line]);
      else if (e.kind === "summary") setSummary(e.payload);
      else if (e.kind === "error") setLines((l) => [...l, `ERROR: ${e.message}`]);
      else if (e.kind === "done") {
        setRunning(false);
        onSummary();
      }
    });
    setRunning(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
      className="glass rounded-2xl border border-white/10 overflow-hidden"
    >
      <div className="p-6 md:p-8 border-b border-white/5 flex items-center gap-2">
        <Cpu className="h-4 w-4 text-ember-400" />
        <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
          Retrain Pipeline
        </div>
      </div>
      <div className="p-6 md:p-8 space-y-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="password"
            placeholder="X-Ops-Token (leave blank if not set)"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm font-mono text-bone-100 placeholder:text-bone-600 focus:outline-none focus:border-ember-400/60"
          />
          <button
            onClick={run}
            disabled={running}
            className="inline-flex items-center gap-2 rounded-lg bg-ember-400 text-ink-900 px-4 py-2 text-xs uppercase tracking-[0.22em] font-mono disabled:opacity-40 disabled:cursor-not-allowed hover:bg-ember-300 transition-colors"
          >
            <Play className="h-3.5 w-3.5" />
            {running ? "training…" : "retrain"}
          </button>
        </div>

        <div
          ref={logRef}
          className="h-56 bg-ink-900/60 border border-white/5 rounded-lg p-3 overflow-y-auto font-mono text-[11px] text-bone-300 whitespace-pre-wrap"
        >
          {lines.length === 0 ? (
            <span className="text-bone-600">logs will stream here…</span>
          ) : (
            lines.map((l, i) => <div key={i}>{l}</div>)
          )}
        </div>

        {summary && summary.event === "done" && (
          <div className="rounded-lg border border-white/10 bg-white/5 p-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.22em] text-bone-400 mb-2">
              Run summary · {summary.promoted ? "promoted" : "rejected"}
            </div>
            <MetricDiff label="F1" before={summary.old_metrics?.f1} after={summary.new_metrics?.f1} />
            <MetricDiff label="Precision" before={summary.old_metrics?.precision} after={summary.new_metrics?.precision} />
            <MetricDiff label="Recall" before={summary.old_metrics?.recall} after={summary.new_metrics?.recall} />
            <MetricDiff label="ROC-AUC" before={summary.old_metrics?.roc_auc} after={summary.new_metrics?.roc_auc} />
            {summary.new_version && (
              <div className="mt-2 text-[10px] font-mono text-bone-500">new version: {summary.new_version}</div>
            )}
          </div>
        )}
        {summary && summary.event === "error" && (
          <div className="rounded-lg border border-signal-red/40 bg-signal-red/5 p-4 text-sm text-signal-red font-mono">
            {summary.error}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function ProfileTabs({
  selected,
  onSelect,
  available,
}: {
  selected: ProfileName;
  onSelect: (p: ProfileName) => void;
  available: AllModelInfo | null;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {PROFILE_ORDER.map((p) => {
        const has = !!available?.[p]?.version;
        const isActive = selected === p;
        return (
          <button
            key={p}
            onClick={() => onSelect(p)}
            className={`px-3.5 py-1.5 rounded-lg text-[11px] uppercase tracking-[0.22em] font-mono transition-colors border ${
              isActive
                ? "bg-ember-400 text-ink-900 border-ember-400"
                : "text-bone-300 border-white/10 hover:border-ember-400/40 hover:text-ember-300"
            }`}
          >
            {PROFILE_LABELS[p] ?? p}
            {!has && (
              <span className={`ml-2 text-[9px] ${isActive ? "text-ink-900/60" : "text-bone-600"}`}>
                · untrained
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function Ops() {
  const [allInfo, setAllInfo] = useState<AllModelInfo | null>(null);
  const [profile, setProfile] = useState<ProfileName>("vehicle_maintenance");
  const [drift, setDrift] = useState<DriftReport | null>(null);

  const refresh = async () => {
    try { setAllInfo(await fetchAllModelInfo()); } catch { /* noop */ }
    try { setDrift(await fetchDrift("7d", profile)); } catch { /* noop */ }
  };

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    setDrift(null);
    fetchDrift("7d", profile).then(setDrift).catch(() => {});
  }, [profile]);

  const info: ModelInfo | null = allInfo?.[profile] ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-bone-400 text-[11px] font-mono uppercase tracking-[0.22em]">
        <Database className="h-3.5 w-3.5" /> MLOps Console
      </div>
      <ProfileTabs selected={profile} onSelect={setProfile} available={allInfo} />
      <RegistryCard info={info} />
      <BaselinesTable info={info} />
      <DriftCard report={drift} />
      <RetrainPanel onSummary={refresh} />
    </div>
  );
}
