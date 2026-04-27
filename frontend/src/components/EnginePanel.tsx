import { SensorPanel, type SensorPanelConfig } from "./SensorPanel";
import { ENGINE_RANGES } from "../lib/anomalyRanges";

export interface EngineInputs {
  engine_rpm: number;
  lub_oil_pressure: number;
  fuel_pressure: number;
  coolant_pressure: number;
  lub_oil_temp: number;
  coolant_temp: number;
}

const CONFIG: SensorPanelConfig<EngineInputs> = {
  title: "Engine · Sensor Readings",
  ranges: ENGINE_RANGES,
  rangeKeys: [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
  ],
  gridCols: 2,
  buttonLabel: "Run engine diagnosis",
  sample: {
    engine_rpm: ENGINE_RANGES.engine_rpm.sample,
    lub_oil_pressure: ENGINE_RANGES.lub_oil_pressure.sample,
    fuel_pressure: ENGINE_RANGES.fuel_pressure.sample,
    coolant_pressure: ENGINE_RANGES.coolant_pressure.sample,
    lub_oil_temp: ENGINE_RANGES.lub_oil_temp.sample,
    coolant_temp: ENGINE_RANGES.coolant_temp.sample,
  },
};

interface Props {
  onPredict: (v: EngineInputs) => void;
  predicting: boolean;
}

export function EnginePanel({ onPredict, predicting }: Props) {
  return <SensorPanel config={CONFIG} onPredict={onPredict} predicting={predicting} />;
}
