// Healthy operating bands for sensor inputs. Used to render slider zones
// AND to produce natural-language commentary alongside the model verdict.

export type Verdict = "ok" | "watch" | "alert";

export interface Range {
  key: string;
  label: string;
  unit: string;
  min: number;          // slider min
  max: number;          // slider max
  step: number;
  okLow: number;        // healthy band lower bound
  okHigh: number;       // healthy band upper bound
  sample: number;       // realistic sample value
  describeLow: string;  // sentence used when below okLow
  describeHigh: string; // sentence used when above okHigh
  describeOk: string;   // short phrase used when inside the band
}

export const HYUNDAI_RANGES: Record<string, Range> = {
  engine_temperature: {
    key: "engine_temperature",
    label: "Engine Temperature",
    unit: "°C",
    min: 40, max: 130, step: 0.5,
    okLow: 80, okHigh: 105,
    sample: 92,
    describeLow: "engine is running cooler than the normal operating range — likely not yet warmed up",
    describeHigh: "engine is running hot and may be approaching overheating — check coolant level and the radiator",
    describeOk: "engine temperature is in the healthy operating band",
  },
  brake_pad_thickness: {
    key: "brake_pad_thickness",
    label: "Brake Pad Thickness",
    unit: "mm",
    min: 0, max: 14, step: 0.1,
    okLow: 6, okHigh: 14,
    sample: 8.2,
    describeLow: "brake pads are worn thin and should be replaced soon",
    describeHigh: "brake pad thickness is unusually high",
    describeOk: "brake pads have plenty of material left",
  },
  tire_pressure: {
    key: "tire_pressure",
    label: "Tire Pressure",
    unit: "PSI",
    min: 18, max: 50, step: 0.1,
    okLow: 30, okHigh: 36,
    sample: 32,
    describeLow: "tire pressure is below the recommended range — under-inflation hurts handling and fuel economy",
    describeHigh: "tire pressure is above the recommended range — over-inflation reduces grip and accelerates center-tread wear",
    describeOk: "tire pressure is within the recommended range",
  },
};

export const ENGINE_RANGES: Record<string, Range> = {
  engine_rpm: {
    key: "engine_rpm",
    label: "Engine RPM",
    unit: "rpm",
    min: 0, max: 6500, step: 10,
    okLow: 700, okHigh: 4500,
    sample: 2200,
    describeLow: "RPM is below idle — engine may be stalling",
    describeHigh: "RPM is very high — sustained operation here stresses the engine",
    describeOk: "RPM is in the typical operating range",
  },
  lub_oil_pressure: {
    key: "lub_oil_pressure",
    label: "Lub Oil Pressure",
    unit: "bar",
    min: 0, max: 8, step: 0.01,
    okLow: 2, okHigh: 5,
    sample: 3.2,
    describeLow: "oil pressure is low — possible oil starvation, check level and pump",
    describeHigh: "oil pressure is high — possible blocked passage or wrong-grade oil",
    describeOk: "oil pressure looks healthy",
  },
  fuel_pressure: {
    key: "fuel_pressure",
    label: "Fuel Pressure",
    unit: "psi",
    min: 0, max: 80, step: 0.1,
    okLow: 30, okHigh: 60,
    sample: 45,
    describeLow: "fuel pressure is low — possible failing pump or clogged filter",
    describeHigh: "fuel pressure is high — possible regulator failure",
    describeOk: "fuel pressure is in spec",
  },
  coolant_pressure: {
    key: "coolant_pressure",
    label: "Coolant Pressure",
    unit: "bar",
    min: 0, max: 5, step: 0.01,
    okLow: 0.8, okHigh: 2.0,
    sample: 1.2,
    describeLow: "coolant pressure is low — check for leaks or a stuck-open cap",
    describeHigh: "coolant pressure is high — possible blockage or thermostat issue",
    describeOk: "coolant pressure is normal",
  },
  lub_oil_temp: {
    key: "lub_oil_temp",
    label: "Lub Oil Temp",
    unit: "°C",
    min: 30, max: 140, step: 0.1,
    okLow: 70, okHigh: 105,
    sample: 85,
    describeLow: "oil is below normal operating temperature — engine likely not warmed up",
    describeHigh: "oil temperature is elevated — extended operation will accelerate oil breakdown",
    describeOk: "oil temperature is in the healthy band",
  },
  coolant_temp: {
    key: "coolant_temp",
    label: "Coolant Temp",
    unit: "°C",
    min: 30, max: 130, step: 0.1,
    okLow: 80, okHigh: 100,
    sample: 88,
    describeLow: "coolant is below normal operating temperature",
    describeHigh: "coolant is hotter than normal — investigate cooling-system performance",
    describeOk: "coolant temperature is in the healthy band",
  },
};

export function classify(r: Range, value: number): Verdict {
  if (value < r.okLow || value > r.okHigh) {
    const span = r.okHigh - r.okLow || 1;
    const dist = value < r.okLow ? r.okLow - value : value - r.okHigh;
    return dist / span > 0.25 ? "alert" : "watch";
  }
  return "ok";
}

export interface FeatureReading {
  range: Range;
  value: number;
  verdict: Verdict;
  sentence: string;
}

export function readFeatures(
  values: Record<string, number>,
  ranges: Record<string, Range>
): FeatureReading[] {
  return Object.values(ranges).map((r) => {
    const v = values[r.key];
    const verdict = classify(r, v);
    let sentence: string;
    if (verdict === "ok") {
      sentence = `${r.label} (${v}${r.unit}) — ${r.describeOk}.`;
    } else if (v < r.okLow) {
      sentence = `${r.label} (${v}${r.unit}) — ${r.describeLow}.`;
    } else {
      sentence = `${r.label} (${v}${r.unit}) — ${r.describeHigh}.`;
    }
    return { range: r, value: v, verdict, sentence };
  });
}

export interface Interpretation {
  headline: string;
  summary: string;
  features: FeatureReading[];
}

// Combine the model's verdict with our per-feature reading into a richer
// recommendation, e.g. "Brake pads and tire pressure look fine, but engine
// temperature is high — investigate cooling before driving."
export function interpret(
  values: Record<string, number>,
  ranges: Record<string, Range>,
  modelStatus: string | null
): Interpretation {
  const features = readFeatures(values, ranges);
  const alerts = features.filter((f) => f.verdict === "alert");
  const watch = features.filter((f) => f.verdict === "watch");
  const ok = features.filter((f) => f.verdict === "ok");

  const flagged = [...alerts, ...watch];
  const okNames = ok.map((f) => f.range.label.toLowerCase()).join(", ");
  const flaggedNames = flagged.map((f) => f.range.label.toLowerCase()).join(", ");

  const anomaly =
    typeof modelStatus === "string" && /anomaly|fault|warning|required|attention/i.test(modelStatus);

  let headline: string;
  let summary: string;

  if (flagged.length === 0) {
    headline = anomaly ? "Model flagged anomaly · sensors look in-band" : "All sensors nominal";
    summary = anomaly
      ? "The model flagged an anomaly even though every input is inside its healthy band. This usually means a subtle multi-feature pattern — log a maintenance check to be safe."
      : `${capitalizeFirst(okNames)} are all reading within their normal ranges. No action needed.`;
  } else if (flagged.length === 1) {
    const f = flagged[0];
    headline = `Pay attention to ${f.range.label.toLowerCase()}`;
    summary = ok.length
      ? `${capitalizeFirst(okNames)} look healthy, but ${f.sentence.toLowerCase()}`
      : capitalizeFirst(f.sentence);
  } else {
    headline = `Multiple readings out of band: ${flaggedNames}`;
    summary = ok.length
      ? `${capitalizeFirst(okNames)} look fine, but ${flagged.length} other readings are outside their healthy bands — see details below.`
      : `${flagged.length} readings are outside their healthy bands — see details below.`;
  }

  return { headline, summary, features };
}

function capitalizeFirst(s: string) {
  return s.length === 0 ? s : s[0].toUpperCase() + s.slice(1);
}
