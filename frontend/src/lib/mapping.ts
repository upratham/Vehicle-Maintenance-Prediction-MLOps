import type { DecodedVin } from "./nhtsa";

export type VehicleCategory = "Car" | "SUV" | "Truck" | "Van" | "Bus" | "Motorcycle";
export type FuelType = "Petrol" | "Diesel" | "Electric";
export type Transmission = "Automatic" | "Manual";

export interface AutoFilled {
  Year: number | null;
  Make: string;
  Model: string;
  Vehicle_Age: number | null;
  Engine_Size: number | null;
  Vehicle_Model: VehicleCategory | null;
  Fuel_Type: FuelType | null;
  Transmission_Type: Transmission | null;
}

const CURRENT_YEAR = new Date().getFullYear();

export function mapBodyClass(body: string, vtype: string): VehicleCategory | null {
  const s = `${body} ${vtype}`.toLowerCase();
  if (/motorcycle|motorbike|scooter/.test(s)) return "Motorcycle";
  if (/bus|coach/.test(s)) return "Bus";
  if (/van|minivan|cargo van/.test(s)) return "Van";
  if (/pickup|truck|tractor/.test(s)) return "Truck";
  if (/sport utility|suv|crossover|multi-purpose/.test(s)) return "SUV";
  if (/sedan|coupe|hatchback|convertible|wagon|roadster|car/.test(s)) return "Car";
  return null;
}

export function mapFuel(f: string): FuelType | null {
  const s = f?.toLowerCase() ?? "";
  if (!s) return null;
  if (/electric|bev/.test(s) && !/hybrid/.test(s)) return "Electric";
  if (/diesel/.test(s)) return "Diesel";
  if (/gas|petrol|gasoline|ethanol|flex/.test(s)) return "Petrol";
  return null;
}

export function mapTransmission(t: string): Transmission | null {
  const s = t?.toLowerCase() ?? "";
  if (!s) return null;
  if (/manual/.test(s)) return "Manual";
  if (/auto|cvt|dct|dsg/.test(s)) return "Automatic";
  return null;
}

export function fromVin(d: DecodedVin): AutoFilled {
  const year = Number(d.ModelYear) || null;
  const cc = Number(d.DisplacementCC);
  const liters = Number(d.DisplacementL);
  const engineCc = Number.isFinite(cc) && cc > 0
    ? Math.round(cc)
    : Number.isFinite(liters) && liters > 0
      ? Math.round(liters * 1000)
      : null;
  return {
    Year: year,
    Make: d.Make ?? "",
    Model: d.Model ?? "",
    Vehicle_Age: year ? Math.max(0, CURRENT_YEAR - year) : null,
    Engine_Size: engineCc,
    Vehicle_Model: mapBodyClass(d.BodyClass ?? "", d.VehicleType ?? ""),
    Fuel_Type: mapFuel(d.FuelTypePrimary ?? ""),
    Transmission_Type: mapTransmission(d.TransmissionStyle ?? ""),
  };
}
