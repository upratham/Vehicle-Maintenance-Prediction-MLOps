import { SensorPanel, type SensorPanelConfig } from "./SensorPanel";
import { HYUNDAI_RANGES } from "../lib/anomalyRanges";

const MAINTENANCE_TYPES = ["Repair", "Routine Maintenance", "Component Replacement"] as const;

export interface HyundaiInputs {
  engine_temperature: number;
  brake_pad_thickness: number;
  tire_pressure: number;
  maintenance_type: string;
}

const CONFIG: SensorPanelConfig<HyundaiInputs> = {
  title: "Vehicle · Sensor Readings",
  ranges: HYUNDAI_RANGES,
  rangeKeys: ["engine_temperature", "brake_pad_thickness", "tire_pressure"],
  gridCols: 1,
  selects: [
    { key: "maintenance_type", label: "Maintenance Type", options: MAINTENANCE_TYPES },
  ],
  buttonLabel: "Run anomaly detection",
  sample: {
    engine_temperature: HYUNDAI_RANGES.engine_temperature.sample,
    brake_pad_thickness: HYUNDAI_RANGES.brake_pad_thickness.sample,
    tire_pressure: HYUNDAI_RANGES.tire_pressure.sample,
    maintenance_type: "Routine Maintenance",
  },
};

interface Props {
  onPredict: (v: HyundaiInputs) => void;
  predicting: boolean;
}

export function HyundaiPanel({ onPredict, predicting }: Props) {
  return <SensorPanel config={CONFIG} onPredict={onPredict} predicting={predicting} />;
}
