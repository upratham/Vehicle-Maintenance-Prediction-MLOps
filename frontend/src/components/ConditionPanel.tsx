import { useState } from "react";
import { motion } from "framer-motion";
import { Activity, Loader2 } from "lucide-react";
import type { AutoFilled } from "../lib/mapping";
import { Field } from "./Field";
import { Select } from "./Select";

const CONDITION = ["New", "Good", "Worn Out"] as const;
const BATTERY = ["Strong", "Good", "Weak"] as const;

export interface ConditionValues {
  Odometer_Reading: number;
  Reported_Issues: number;
  Accident_History: number;
  Fuel_Efficiency: number;
  Tire_Condition: string;
  Brake_Condition: string;
  Battery_Status: string;
}

interface Props {
  auto: AutoFilled;
  vin: string | null;
  onPredict: (payload: ConditionValues & { auto: AutoFilled; vin: string | null }) => Promise<void>;
  predicting: boolean;
}

export function ConditionPanel({ auto, vin, onPredict, predicting }: Props) {
  const [odometer, setOdometer] = useState("");
  const [issues, setIssues] = useState("0");
  const [accidents, setAccidents] = useState("0");
  const [efficiency, setEfficiency] = useState("");
  const [tire, setTire] = useState("");
  const [brake, setBrake] = useState("");
  const [battery, setBattery] = useState("");

  const ready = odometer && efficiency && tire && brake && battery;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!ready) return;
    onPredict({
      auto,
      vin,
      Odometer_Reading: Number(odometer),
      Reported_Issues: Number(issues || 0),
      Accident_History: Number(accidents || 0),
      Fuel_Efficiency: Number(efficiency),
      Tire_Condition: tire,
      Brake_Condition: brake,
      Battery_Status: battery,
    });
  };

  return (
    <motion.form
      onSubmit={submit}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay: 0.1 }}
      className="glass rounded-2xl p-7 md:p-9 shadow-inset1"
    >
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2 text-bone-400 text-[11px] uppercase tracking-[0.28em]">
          <Activity className="h-3.5 w-3.5" />
          02 · Vehicle Telemetry
        </div>
        <span className="text-[11px] font-mono text-bone-600">
          {vin ? `VIN ${vin.slice(0, 5)}…${vin.slice(-4)}` : "Manual entry"}
        </span>
      </div>

      {/* Locked identity row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-5 pb-6 mb-6 border-b border-white/5">
        <Field label="Year" value={auto.Year ?? ""} locked />
        <Field label="Make" value={auto.Make} locked />
        <Field label="Model" value={auto.Model} locked />
        <Field label="Category" value={auto.Vehicle_Model ?? "—"} locked />
        <Field label="Engine" value={auto.Engine_Size ?? ""} suffix="cc" locked />
        <Field label="Fuel" value={auto.Fuel_Type ?? "—"} locked />
        <Field label="Transmission" value={auto.Transmission_Type ?? "—"} locked />
        <Field label="Vehicle Age" value={auto.Vehicle_Age ?? ""} suffix="yrs" locked />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-5">
        <Field
          label="Odometer"
          type="number"
          value={odometer}
          onChange={(e) => setOdometer(e.target.value)}
          placeholder="60000"
          suffix="km"
          required
        />
        <Field
          label="Fuel Efficiency"
          type="number"
          step="0.1"
          value={efficiency}
          onChange={(e) => setEfficiency(e.target.value)}
          placeholder="14.5"
          suffix="km/l"
          required
        />
        <Field
          label="Reported Issues"
          type="number"
          value={issues}
          onChange={(e) => setIssues(e.target.value)}
          placeholder="0"
          required
        />
        <Field
          label="Accident History"
          type="number"
          value={accidents}
          onChange={(e) => setAccidents(e.target.value)}
          placeholder="0"
          required
        />
        <Select
          label="Tire Condition"
          value={tire}
          onChange={(e) => setTire(e.target.value)}
          options={CONDITION}
          required
        />
        <Select
          label="Brake Condition"
          value={brake}
          onChange={(e) => setBrake(e.target.value)}
          options={CONDITION}
          required
        />
        <Select
          label="Battery Status"
          value={battery}
          onChange={(e) => setBattery(e.target.value)}
          options={BATTERY}
          required
        />
      </div>

      <div className="mt-7 flex justify-end">
        <button
          type="submit"
          disabled={!ready || predicting}
          className="px-7 py-3 rounded-md bg-ember-400 text-ink-900 font-medium text-sm tracking-wide
                     hover:bg-ember-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed
                     flex items-center gap-2 shadow-glow"
        >
          {predicting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {predicting ? "Analyzing" : "Run Prediction"}
        </button>
      </div>
    </motion.form>
  );
}
