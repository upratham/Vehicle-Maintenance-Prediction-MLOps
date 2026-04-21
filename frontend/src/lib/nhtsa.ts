// NHTSA vPIC API helpers — free, no key required.
// Docs: https://vpic.nhtsa.dot.gov/api/

const BASE = "https://vpic.nhtsa.dot.gov/api/vehicles";

export interface DecodedVin {
  VIN: string;
  ModelYear: string;
  Make: string;
  Model: string;
  BodyClass: string;
  VehicleType: string;
  FuelTypePrimary: string;
  DisplacementCC: string;
  DisplacementL: string;
  TransmissionStyle: string;
  EngineCylinders: string;
  ErrorCode: string;
  ErrorText: string;
}

export async function decodeVin(vin: string): Promise<DecodedVin> {
  const url = `${BASE}/DecodeVinValues/${encodeURIComponent(vin)}?format=json`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`NHTSA decode failed (${r.status})`);
  const j = await r.json();
  const row = j?.Results?.[0];
  if (!row) throw new Error("Empty NHTSA response");
  return row as DecodedVin;
}

export async function getAllMakes(): Promise<string[]> {
  const r = await fetch(`${BASE}/GetAllMakes?format=json`);
  const j = await r.json();
  return (j?.Results ?? [])
    .map((x: { Make_Name: string }) => x.Make_Name)
    .filter(Boolean)
    .sort();
}

// Curated list of common consumer vehicle makes. The full NHTSA list has
// thousands of niche/industrial manufacturers that clutter the dropdown.
export const POPULAR_MAKES: string[] = [
  "Acura", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler",
  "Dodge", "Ferrari", "Fiat", "Ford", "Genesis", "GMC", "Honda", "Hyundai",
  "Infiniti", "Jaguar", "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln",
  "Maserati", "Mazda", "Mercedes-Benz", "Mini", "Mitsubishi", "Nissan",
  "Porsche", "Ram", "Subaru", "Tesla", "Toyota", "Volkswagen", "Volvo",
];

export async function getModelsForMakeYear(
  make: string,
  year: number | string
): Promise<string[]> {
  const url = `${BASE}/GetModelsForMakeYear/make/${encodeURIComponent(
    make
  )}/modelyear/${year}?format=json`;
  const r = await fetch(url);
  const j = await r.json();
  return (j?.Results ?? [])
    .map((x: { Model_Name: string }) => x.Model_Name)
    .filter(Boolean)
    .sort();
}
