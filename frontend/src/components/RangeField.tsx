import { useMemo } from "react";
import { classify, type Range as SensorRange, type Verdict } from "../lib/anomalyRanges";

interface Props {
  range: SensorRange;
  value: number;
  onChange: (n: number) => void;
}

const verdictColor: Record<Verdict, string> = {
  ok: "text-signal-green",
  watch: "text-ember-300",
  alert: "text-signal-red",
};

export function RangeField({ range, value, onChange }: Props) {
  const verdict = classify(range, value);

  const { okLeftPct, okWidthPct } = useMemo(() => {
    const span = range.max - range.min;
    return {
      okLeftPct: ((range.okLow - range.min) / span) * 100,
      okWidthPct: ((range.okHigh - range.okLow) / span) * 100,
    };
  }, [range]);

  return (
    <div className="block">
      <div className="flex items-baseline justify-between gap-3">
        <span className="field-label">{range.label}</span>
        <div className="flex items-baseline gap-1">
          <input
            type="number"
            min={range.min}
            max={range.max}
            step={range.step}
            value={value}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (!Number.isNaN(n)) onChange(n);
            }}
            className={`w-20 bg-transparent text-right border-b border-white/10 focus:outline-none focus:border-ember-400 font-mono text-[12px] tabular-nums py-0.5 ${verdictColor[verdict]}`}
          />
          <span className="text-bone-600 font-mono text-[10px]">{range.unit}</span>
        </div>
      </div>

      <div className="relative pt-3 pb-1">
        {/* track + healthy zone */}
        <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-1 rounded-full bg-white/5">
          <div
            className="absolute top-0 bottom-0 rounded-full bg-signal-green/30"
            style={{ left: `${okLeftPct}%`, width: `${okWidthPct}%` }}
          />
        </div>
        <input
          type="range"
          min={range.min}
          max={range.max}
          step={range.step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="relative w-full appearance-none bg-transparent cursor-pointer h-4
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-ember-400
                     [&::-webkit-slider-thumb]:border-2
                     [&::-webkit-slider-thumb]:border-ink-900
                     [&::-webkit-slider-thumb]:shadow-md
                     [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5
                     [&::-moz-range-thumb]:rounded-full
                     [&::-moz-range-thumb]:bg-ember-400
                     [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-ink-900"
        />
      </div>

      <div className="flex items-baseline justify-between text-[10px] font-mono text-bone-600">
        <span>{range.min}{range.unit}</span>
        <span>typical {range.okLow}–{range.okHigh}{range.unit}</span>
        <span>{range.max}{range.unit}</span>
      </div>
    </div>
  );
}
