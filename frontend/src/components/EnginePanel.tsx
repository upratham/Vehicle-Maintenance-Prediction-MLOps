import { useState } from "react";
import { motion } from "framer-motion";
import { RangeField } from "./RangeField";
import { ENGINE_RANGES } from "../lib/anomalyRanges";

export interface EngineInputs {
  engine_rpm: number;
  lub_oil_pressure: number;
  fuel_pressure: number;
  coolant_pressure: number;
  lub_oil_temp: number;
  coolant_temp: number;
}

const SAMPLE: EngineInputs = {
  engine_rpm: ENGINE_RANGES.engine_rpm.sample,
  lub_oil_pressure: ENGINE_RANGES.lub_oil_pressure.sample,
  fuel_pressure: ENGINE_RANGES.fuel_pressure.sample,
  coolant_pressure: ENGINE_RANGES.coolant_pressure.sample,
  lub_oil_temp: ENGINE_RANGES.lub_oil_temp.sample,
  coolant_temp: ENGINE_RANGES.coolant_temp.sample,
};

interface Props {
  onPredict: (v: EngineInputs) => void;
  predicting: boolean;
}

export function EnginePanel({ onPredict, predicting }: Props) {
  const [v, setV] = useState<EngineInputs>(SAMPLE);

  const set = <K extends keyof EngineInputs>(k: K, x: EngineInputs[K]) =>
    setV((prev) => ({ ...prev, [k]: x }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-6 md:p-8 space-y-7"
    >
      <div>
        <div className="text-[11px] uppercase tracking-[0.28em] text-bone-400">
          Engine · Sensor Readings
        </div>
        <div className="text-bone-500 text-xs mt-1">
          Drag the sliders or type values directly. Green band marks the healthy operating range.
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <RangeField range={ENGINE_RANGES.engine_rpm} value={v.engine_rpm} onChange={(n) => set("engine_rpm", n)} />
        <RangeField range={ENGINE_RANGES.lub_oil_pressure} value={v.lub_oil_pressure} onChange={(n) => set("lub_oil_pressure", n)} />
        <RangeField range={ENGINE_RANGES.fuel_pressure} value={v.fuel_pressure} onChange={(n) => set("fuel_pressure", n)} />
        <RangeField range={ENGINE_RANGES.coolant_pressure} value={v.coolant_pressure} onChange={(n) => set("coolant_pressure", n)} />
        <RangeField range={ENGINE_RANGES.lub_oil_temp} value={v.lub_oil_temp} onChange={(n) => set("lub_oil_temp", n)} />
        <RangeField range={ENGINE_RANGES.coolant_temp} value={v.coolant_temp} onChange={(n) => set("coolant_temp", n)} />
      </div>

      <div className="pt-2">
        <button
          onClick={() => onPredict(v)}
          disabled={predicting}
          className="inline-flex items-center gap-2 rounded-lg bg-ember-400 text-ink-900 px-5 py-2.5 text-xs uppercase tracking-[0.22em] font-mono disabled:opacity-40 disabled:cursor-not-allowed hover:bg-ember-300 transition-colors"
        >
          {predicting ? "predicting…" : "Run engine diagnosis"}
        </button>
      </div>
    </motion.div>
  );
}
