# Vehicle Maintenance Prediction

MSML 605 final project. FastAPI + React app, three vehicle-health models.

## How to run

You'll need Python 3.12 and Node 20+. Open a terminal and run:

**macOS / Linux**

```
git clone https://github.com/upratham/Vehicle-Maintenance-Prediction-MLOps
cd Vehicle-Maintenance-Prediction-MLOps
bash .dev/dev.sh
```

**Windows (PowerShell)**

```
git clone https://github.com/upratham/Vehicle-Maintenance-Prediction-MLOps
cd Vehicle-Maintenance-Prediction-MLOps
.\.dev\dev.ps1
```

Then wait.

That's it. The script makes a venv, installs deps, trains the three models
from the CSVs in `data/`, builds the frontend, and starts the backend on
`:8000` and the frontend on `:3000`. First run takes ~5-15 min for training,
after that it's a few seconds.

If PowerShell blocks the script the first time:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## With or without `.env`

You don't need any credentials to run it. Training, predictions, and drift
all work locally with no setup. `.env` just unlocks the cloud-backed parts:

- `CONNECTION_URL` (Mongo Atlas) - real drift logs, artifact registry
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - S3 model push
- `CLEARML_API_*` - Optuna trial tracking
- `OPS_TOKEN` - auth on `/train`

Anything missing just logs a warning and skips. Copy the template if you
want to fill some in:

```
cp .env.example .env
```

## URLs once it's up

- `localhost:3000` predictor UI (3 tabs)
- `localhost:3000/ops` ops console (registry, drift, retrain)
- `localhost:8000/docs` swagger

## Models

vehicle_maintenance (50k rows, Random Forest), cars_hyundai (1.1k rows,
RF or SVM picked by HPO), engine_data (19.5k rows, Random Forest). All
three share the same pipeline: ingest -> validate -> transform -> HPO
train -> evaluate -> push.

To retrain a single profile manually:

```python
from src.pipeline.training_pipeline import TrainPipeline
TrainPipeline().run_pipeline(profile_names=["engine_data"])
```

## Stack

FastAPI, React + Vite + Tailwind, scikit-learn, imbalanced-learn (SMOTE),
XGBoost, Optuna, ClearML, MongoDB Atlas, AWS S3, GitHub Actions.

## Authors

Prathamesh Uravane, Sankeerth B, Claude B
