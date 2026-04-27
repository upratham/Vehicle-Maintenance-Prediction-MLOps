import { useState } from "react";
import { motion } from "framer-motion";
import { Select } from "./Select";
import { RangeField } from "./RangeField";
import type { Range } from "../lib/anomalyRanges";

export interface SensorPanelConfig<V extends Record<string, number | string>> {
  title: string;
  ranges: Record<string, Range>;
  rangeKeys: (keyof V)[];
  gridCols?: 1 | 2;
  selects?: Array<{ key: keyof V; label: string; options: readonly string[] }>;
  buttonLabel: string;
  sample: V;
}

interface Props<V extends Record<string, number | string>> {
  config: SensorPanelConfig<V>;
  onPredict: (v: V) => void;
  predicting: boolean;
}

export function SensorPanel<V extends Record<string, number | string>>({
  config,
  onPredict,
  predicting,
}: Props<V>) {
  const [v, setV] = useState<V>(config.sample);

  const set = <K extends keyof V>(k: K, x: V[K]) =>
    setV((prev) => ({ ...prev, [k]: x }));

  const gridClass = config.gridCols === 2 ? "grid gap-6 md:grid-cols-2" : "grid gap-6";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-6 md:p-8 space-y-7"
    >
      <div>
        <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
          {config.title}
        </div>
        <div className="text-bone-500 text-xs mt-1">
          Drag the sliders or type values directly. Green band marks the healthy operating range.
        </div>
      </div>

      <div className={gridClass}>
        {config.rangeKeys.map((key) => (
          <RangeField
            key={String(key)}
            range={config.ranges[String(key)]}
            value={v[key] as number}
            onChange={(n) => set(key, n as V[typeof key])}
          />
        ))}
      </div>

      {config.selects && config.selects.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {config.selects.map(({ key, label, options }) => (
            <Select
              key={String(key)}
              label={label}
              options={options}
              value={v[key] as string}
              onChange={(e) => set(key, e.target.value as V[typeof key])}
            />
          ))}
        </div>
      )}

      <div className="pt-2">
        <button
          onClick={() => onPredict(v)}
          disabled={predicting}
          className="inline-flex items-center gap-2 rounded-lg bg-ember-400 text-ink-900 px-5 py-2.5 text-xs uppercase tracking-[0.22em] font-mono disabled:opacity-40 disabled:cursor-not-allowed hover:bg-ember-300 transition-colors"
        >
          {predicting ? "predicting…" : config.buttonLabel}
        </button>
      </div>
    </motion.div>
  );
}
