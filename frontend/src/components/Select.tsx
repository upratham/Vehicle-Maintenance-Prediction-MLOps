import { ChevronDown } from "lucide-react";
import { cn } from "../lib/utils";

interface Props extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
  options: readonly string[];
}

export function Select({ label, options, className, ...rest }: Props) {
  return (
    <label className="block">
      <span className="field-label">{label}</span>
      <div className="relative">
        <select
          {...rest}
          className={cn(
            "field-input appearance-none pr-6 font-mono",
            "[color-scheme:dark]",
            className
          )}
        >
          <option value="" disabled>
            Select…
          </option>
          {options.map((o) => (
            <option key={o} value={o} className="bg-ink-800 text-bone-50">
              {o}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 h-4 w-4 text-bone-400" />
      </div>
    </label>
  );
}
