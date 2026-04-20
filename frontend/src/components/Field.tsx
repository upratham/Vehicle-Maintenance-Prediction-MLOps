import { Lock } from "lucide-react";
import { cn } from "../lib/utils";

interface Props extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  locked?: boolean;
  suffix?: string;
}

export function Field({ label, locked, suffix, className, ...rest }: Props) {
  return (
    <label className="block">
      <span className="field-label flex items-center gap-1.5">
        {label}
        {locked && <Lock className="h-2.5 w-2.5 text-ember-400" />}
      </span>
      <div className="relative flex items-baseline gap-2">
        <input
          {...rest}
          readOnly={locked || rest.readOnly}
          className={cn("field-input", className)}
        />
        {suffix && (
          <span className="absolute right-0 bottom-2.5 text-[10px] uppercase tracking-widest text-bone-600 pointer-events-none">
            {suffix}
          </span>
        )}
      </div>
    </label>
  );
}
