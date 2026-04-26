import { useState } from "react";
import { motion } from "framer-motion";
import { Select } from "./Select";
import { RangeField } from "./RangeField";
import { HYUNDAI_RANGES } from "../lib/anomalyRanges";

const MAINTENANCE_TYPES = ["Repair", "Routine Maintenance", "Component Replacement"] as const;

export interface HyundaiInputs {
  engine_temperature: number;
  brake_pad_thickness: number;
  tire_pressure: number;
  maintenance_type: string;
}

const SAMPLE: HyundaiInputs = {
  engine_temperature: HYUNDAI_RANGES.engine_temperature.sample,
  brake_pad_thickness: HYUNDAI_RANGES.brake_pad_thickness.sample,
  tire_pressure: HYUNDAI_RANGES.tire_pressure.sample,
  maintenance_type: "Routine Maintenance",
};

interface Props {
  onPredict: (v: HyundaiInputs) => void;
  predicting: boolean;
}

export function HyundaiPanel({ onPredict, predicting }: Props) {
  const [v, setV] = useState<HyundaiInputs>(SAMPLE);

  const set = <K extends keyof HyundaiInputs>(k: K, x: HyundaiInputs[K]) =>
    setV((prev) => ({ ...prev, [k]: x }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-6 md:p-8 space-y-7"
    >
      <div>
        <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
          Hyundai · Sensor Readings
        </div>
        <div className="text-bone-500 text-xs mt-1">
          Drag the sliders or type values directly. Green band marks the healthy operating range.
        </div>
      </div>

      <div className="grid gap-6">
        <RangeField
          range={HYUNDAI_RANGES.engine_temperature}
          value={v.engine_temperature}
          onChange={(n) => set("engine_temperature", n)}
        />
        <RangeField
          range={HYUNDAI_RANGES.brake_pad_thickness}
          value={v.brake_pad_thickness}
          onChange={(n) => set("brake_pad_thickness", n)}
        />
        <RangeField
          range={HYUNDAI_RANGES.tire_pressure}
          value={v.tire_pressure}
          onChange={(n) => set("tire_pressure", n)}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Select
          label="Maintenance Type"
          options={MAINTENANCE_TYPES}
          value={v.maintenance_type}
          onChange={(e) => set("maintenance_type", e.target.value)}
        />
      </div>

      <div className="pt-2">
        <button
          onClick={() => onPredict(v)}
          disabled={predicting}
          className="inline-flex items-center gap-2 rounded-lg bg-ember-400 text-ink-900 px-5 py-2.5 text-xs uppercase tracking-[0.22em] font-mono disabled:opacity-40 disabled:cursor-not-allowed hover:bg-ember-300 transition-colors"
        >
          {predicting ? "predicting…" : "Run anomaly detection"}
        </button>
      </div>
    </motion.div>
  );
}
