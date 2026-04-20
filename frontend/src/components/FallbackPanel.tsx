import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ListFilter, Loader2 } from "lucide-react";
import { POPULAR_MAKES, getModelsForMakeYear } from "../lib/nhtsa";
import type { AutoFilled, VehicleCategory, FuelType, Transmission } from "../lib/mapping";
import { Select } from "./Select";

const CURRENT_YEAR = new Date().getFullYear();
const YEARS = Array.from({ length: 35 }, (_, i) => String(CURRENT_YEAR - i));
const CATEGORIES: VehicleCategory[] = ["Car", "SUV", "Truck", "Van", "Bus", "Motorcycle"];
const FUELS: FuelType[] = ["Petrol", "Diesel", "Electric"];
const TRANS: Transmission[] = ["Automatic", "Manual"];

interface Props {
  onSubmit: (a: AutoFilled) => void;
}

export function FallbackPanel({ onSubmit }: Props) {
  const [year, setYear] = useState("");
  const [make, setMake] = useState("");
  const [model, setModel] = useState("");
  const [engineCc, setEngineCc] = useState("");
  const [category, setCategory] = useState<VehicleCategory | "">("");
  const [fuel, setFuel] = useState<FuelType | "">("");
  const [trans, setTrans] = useState<Transmission | "">("");
  const [models, setModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    setModel("");
    setModels([]);
    if (make && year) {
      setLoadingModels(true);
      getModelsForMakeYear(make, year)
        .then(setModels)
        .catch(() => setModels([]))
        .finally(() => setLoadingModels(false));
    }
  }, [make, year]);

  const canSubmit = year && make && model && engineCc && category && fuel && trans;

  const handle = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      Year: Number(year),
      Make: make,
      Model: model,
      Vehicle_Age: Math.max(0, CURRENT_YEAR - Number(year)),
      Engine_Size: Number(engineCc),
      Vehicle_Model: category as VehicleCategory,
      Fuel_Type: fuel as FuelType,
      Transmission_Type: trans as Transmission,
    });
  };

  return (
    <motion.form
      onSubmit={handle}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-6 md:p-7 shadow-inset1"
    >
      <div className="flex items-center gap-2 text-bone-400 text-[11px] uppercase tracking-[0.28em] mb-5">
        <ListFilter className="h-3.5 w-3.5" />
        Manual Entry
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-x-5 gap-y-4">
        <Select label="Year" value={year} onChange={(e) => setYear(e.target.value)} options={YEARS} required />
        <Select
          label="Make"
          value={make}
          onChange={(e) => setMake(e.target.value)}
          options={POPULAR_MAKES}
          required
          disabled={!year}
        />
        <Select
          label={loadingModels ? "Model (loading…)" : "Model"}
          value={model}
          onChange={(e) => setModel(e.target.value)}
          options={models}
          required
          disabled={!make || loadingModels}
        />
        <label className="block">
          <span className="field-label">Engine Size (cc)</span>
          <input
            type="number"
            value={engineCc}
            onChange={(e) => setEngineCc(e.target.value)}
            placeholder="1500"
            className="field-input"
            required
          />
        </label>
        <Select
          label="Category"
          value={category}
          onChange={(e) => setCategory(e.target.value as VehicleCategory)}
          options={CATEGORIES}
          required
        />
        <Select
          label="Fuel Type"
          value={fuel}
          onChange={(e) => setFuel(e.target.value as FuelType)}
          options={FUELS}
          required
        />
        <Select
          label="Transmission"
          value={trans}
          onChange={(e) => setTrans(e.target.value as Transmission)}
          options={TRANS}
          required
        />
      </div>

      <div className="mt-5 flex justify-end">
        <button
          type="submit"
          disabled={!canSubmit}
          className="px-6 py-2.5 rounded-md bg-ember-400 text-ink-900 font-medium text-sm tracking-wide
                     hover:bg-ember-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed
                     flex items-center gap-2"
        >
          {loadingModels ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          Continue
        </button>
      </div>
    </motion.form>
  );
}
