# Frontend — Vehicle Maintenance Predictor

React + TypeScript + Vite + Tailwind + Framer Motion SPA for the MSML605 final project.

## Flow
1. **VIN mode** (default): enter a 17-char VIN → NHTSA vPIC decodes it → auto-fills & **locks** Year, Make, Model, Engine Size, Fuel, Transmission, Vehicle Category, Vehicle Age.
2. **Manual mode** (fallback): cascading Year → Make → Model dropdowns (also from NHTSA) plus engine/fuel/trans/category.
3. **Telemetry**: user fills only what the model needs that can't be auto-derived — Odometer, Fuel Efficiency, Reported Issues, Accidents, Tire/Brake/Battery condition.
4. **POST `/predict`** to FastAPI (proxied in dev) → animated result card.

## Run

```bash
# terminal 1 — backend
cd ..
python app.py           # FastAPI on :5000 (includes /predict JSON endpoint)

# terminal 2 — frontend
cd frontend
npm install
npm run dev             # Vite dev server on :5173, proxies /predict → :5000
```

Build for production: `npm run build` → `dist/`.

## Files
```
src/
  App.tsx                 # page-level state machine
  lib/
    nhtsa.ts              # vPIC client
    mapping.ts            # NHTSA → model-input mapping
    utils.ts              # cn()
  components/
    VinPanel.tsx          # step 01 — VIN decode
    FallbackPanel.tsx     # step 01 — manual Y/M/M
    ConditionPanel.tsx    # step 02 — telemetry + locked identity
    ResultCard.tsx        # animated prediction output
    Field.tsx, Select.tsx # styled primitives
```

## Notes
- NHTSA vPIC: free, no key — `DecodeVinValues`, `GetAllMakes`, `GetModelsForMakeYear`.
- Tailwind v3 + custom tokens in `tailwind.config.js`.
- Aesthetic: dark automotive-editorial — Fraunces serif display, Geist body, JetBrains Mono for data readouts, amber warning-light accent.
