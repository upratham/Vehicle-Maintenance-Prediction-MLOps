# Vehicle Maintenance Prediction

MSML 605 final project. FastAPI + React app that trains and serves three
vehicle-health classifiers.

## Run it

Need Python 3.12 and Node 20+. From the repo root:

```bash
bash .dev/dev.sh        # mac / linux
.\.dev\dev.ps1          # windows powershell
```

Makes a venv, installs deps, trains the three models from the CSVs in
`data/`, builds the frontend, starts backend on `:8000` and frontend on
`:3000`. First run is ~5-15 min for training, then it boots in seconds.

If PowerShell blocks the script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## With or without `.env`

You don't need any credentials. Training, predictions, and drift all work
locally with no setup. `.env` just unlocks the cloud parts:

- `CONNECTION_URL` (Mongo Atlas) - real drift logs, artifact registry
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - S3 model push
- `CLEARML_API_*` - Optuna trial tracking
- `OPS_TOKEN` - auth on `/train`

Anything missing just logs a warning and skips. Copy from the template:

```bash
cp .env.example .env
```

## Endpoints

- `localhost:3000` - predictor UI (3 tabs)
- `localhost:3000/ops` - ops console
- `localhost:8000/docs` - swagger

POST `/predict`, `/predict/hyundai`, `/predict/engine`. GET `/model_info`,
`/drift`. POST `/train` retrains and streams logs over SSE.

## Models

| profile | rows | model |
|---|---|---|
| vehicle_maintenance | 50,000 | Random Forest |
| cars_hyundai | 1,100 | RF or SVM (HPO picks) |
| engine_data | 19,535 | Random Forest |

Pipeline: ingest -> validate -> transform -> HPO -> evaluate -> push.

To train one profile:

```python
from src.pipeline.training_pipeline import TrainPipeline
TrainPipeline().run_pipeline(profile_names=["engine_data"])
```

## Stack

FastAPI, React + Vite + Tailwind, scikit-learn, imbalanced-learn (SMOTE),
XGBoost, Optuna, ClearML, MongoDB Atlas, AWS S3, GitHub Actions.

## Authors

Prathamesh Uravane, Sankeerth B, Claude B
