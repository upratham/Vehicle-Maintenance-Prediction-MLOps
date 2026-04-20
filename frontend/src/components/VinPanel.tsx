import { useState } from "react";
import { motion } from "framer-motion";
import { ScanLine, AlertTriangle, Loader2 } from "lucide-react";
import { decodeVin } from "../lib/nhtsa";
import { fromVin, type AutoFilled } from "../lib/mapping";

interface Props {
  onDecoded: (a: AutoFilled, vin: string) => void;
}

export function VinPanel({ onDecoded }: Props) {
  const [vin, setVin] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const clean = vin.trim().toUpperCase();
    if (clean.length < 11) {
      setErr("VIN should be 17 characters (11+ accepted for partial decode).");
      return;
    }
    setErr(null);
    setLoading(true);
    try {
      const decoded = await decodeVin(clean);
      if (!decoded.Make && !decoded.ModelYear) {
        throw new Error("VIN not recognized. Check and try again.");
      }
      onDecoded(fromVin(decoded), clean);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Decode failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.form
      onSubmit={submit}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.2, 0.8, 0.2, 1] }}
      className="glass rounded-2xl p-7 md:p-9 shadow-inset1"
    >
      <div className="flex items-center gap-2 text-bone-400 text-[11px] uppercase tracking-[0.28em] mb-5">
        <ScanLine className="h-3.5 w-3.5" />
        Identify Vehicle
      </div>

      <label className="field-label">Vehicle Identification Number</label>
      <div className="flex items-stretch gap-3 mt-1">
        <input
          value={vin}
          onChange={(e) => setVin(e.target.value.toUpperCase())}
          placeholder="1HGCM82633A123456"
          maxLength={17}
          className="field-input !text-lg !py-3 tracking-[0.18em]"
          autoFocus
        />
        <button
          type="submit"
          disabled={loading}
          className="shrink-0 px-5 rounded-md bg-ember-400 text-ink-900 font-medium text-sm tracking-wide
                     hover:bg-ember-300 transition-colors disabled:opacity-60 disabled:cursor-not-allowed
                     flex items-center gap-2"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {loading ? "Decoding" : "Decode"}
        </button>
      </div>

      <div className="mt-3 text-[11px] text-bone-600 font-mono">
        <span>{vin.length}/17</span>
      </div>

      {err && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 flex items-center gap-2 text-signal-red text-xs"
        >
          <AlertTriangle className="h-3.5 w-3.5" />
          {err}
        </motion.div>
      )}
    </motion.form>
  );
}
