# Vehicle Maintenance Prediction

End-to-end MLOps system that trains, evaluates, and serves three independent vehicle-health classifiers — one per dataset — through a single unified pipeline. Every model has its own transformer, its own Optuna HPO run logged to ClearML, and its own versioned entry in an S3 model registry.

**Live demo:** https://vehicle-maintenance-prediction-mlops.onrender.com
**Course:** MSML 605 Final Project, University of Maryland.

## Models

| Profile | Dataset | Rows | Target | Model |
|---|---|---|---|---|
| `vehicle_maintenance` | `vehicle_maintenance_data.csv` | 50,000 | `Need_Maintenance` | Random Forest |
| `cars_hyundai` | `cars_hyundai.csv` | 1,100 | `Anomaly Indication` | Decision Tree |
| `engine_data` | `engine_data.csv` | 19,535 | `Engine Condition` | Random Forest |

All three share the same six-stage pipeline: ingest → validate → transform → HPO train → evaluate → push. Each profile uses Optuna for hyperparameter search (20 trials, 3-fold CV F1) with every trial streamed to ClearML.

## Setup

Requires Python 3.12, Node 20+, MongoDB Atlas access, an AWS account with S3 access, and a free [ClearML](https://app.clear.ml) account.

```bash
git clone https://github.com/upratham/Vehicle-Maintenance-Prediction-MLOps.git
cd Vehicle-Maintenance-Prediction-MLOps

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

Copy `.env.example` to `.env` and fill in MongoDB, AWS, and ClearML credentials. The `OPS_TOKEN` value gates the `/train` endpoint — leave empty to allow unauthenticated retraining locally.

## Running

```bash
bash .dev/dev.sh        # backend + frontend together
bash .dev/be.sh         # backend only — http://localhost:8000
bash .dev/fe.sh         # frontend only — http://localhost:3000
```

| URL | Purpose |
|---|---|
| http://localhost:3000 | React predictor with SHAP explanations |
| http://localhost:3000/ops | Model registry, drift, retrain |
| http://localhost:8000 | Server-rendered form (all three models) |
| http://localhost:8000/docs | FastAPI Swagger UI |

To train all three profiles end-to-end:

```bash
python demo.py
```

Local CSVs in `data/` are used by default; MongoDB is only hit when a CSV is missing. To run a single profile or force a Mongo refresh:

```python
from src.pipeline.training_pipeline import TrainPipeline
TrainPipeline().run_pipeline(profile_names=["engine_data"])
TrainPipeline().run_pipeline(refresh_collections=True)
```

Run artifacts land in `artifact/<profile>/<timestamp>/`. Preprocessors are pickled to `preprocessor_obj/<profile>/preprocessing.pkl`.

## Architecture

```
data/<csv> or MongoDB Atlas
        │
        ▼
  Ingestion → Validation → Transformation → HPO Trainer → Evaluation → Pusher
                                                │              │           │
                                                ▼              ▼           ▼
                                            ClearML       compare F1     S3 registry
                                                          vs. deployed   model.pkl +
                                                                         model_registry.json
        ▼
   FastAPI (app.py)
   /predict /model_info /drift /train
        │
        ▼
   React UI  →  Docker  →  ECR  →  EC2 (GitHub Actions)
```

Each pipeline stage is profile-aware. `data_transformation.py` and `model_trainer.py` each define a dispatcher class that selects the correct profile-specific implementation by name (`__new__` pattern), so the orchestrator stays profile-agnostic.

## API

| Route | Method | Description | Auth |
|---|---|---|---|
| `/predict` | POST | Vehicle Maintenance prediction with SHAP attribution | — |
| `/predict/hyundai` | POST | Hyundai anomaly detection | — |
| `/predict/engine` | POST | Engine condition | — |
| `/model_info` | GET | Version, hash, metrics, baselines per profile | — |
| `/drift?window=7d` | GET | Per-feature PSI vs. training distribution | — |
| `/train` | POST | SSE-streamed training; returns old-vs-new metric diff | `X-Ops-Token` |
| `/docs` | GET | Swagger UI | — |

## Project layout

```
app.py                    FastAPI entry point
demo.py                   CLI to run all three training pipelines
config/                   Schemas + per-profile model configs
data/                     CSV cache (vehicle_maintenance, cars_hyundai, engine_data)
preprocessor_obj/         Pickled ColumnTransformer per profile

src/
  components/             Six pipeline stages (ingestion → pusher)
  pipeline/               TrainPipeline + PredictionPipeline
  cloud_storage/          S3 + MongoDB clients
  configuration/          Connection helpers
  entity/                 Config and artifact dataclasses
  constants/              Defaults and HPO parameters
  drift.py                PSI drift computation
  insights.py             SHAP-driven service recommendations

frontend/
  src/App.tsx             Predictor
  src/pages/Ops.tsx       MLOps dashboard

.dev/                     Dev runner scripts
.github/workflows/        Build → ECR → EC2 deploy
```

## Stack

FastAPI + Uvicorn, React 19 + Vite + Tailwind, scikit-learn + imbalanced-learn (SMOTE), XGBoost, Optuna, SHAP, ClearML, MongoDB Atlas, AWS S3 + ECR + EC2, GitHub Actions.

## Authors

| Name | Role |
|---|---|
| Prathamesh Uravane | Deployment & MLOps |
| Sankeerth B | Data & Validation |
| Claude B | Modeling & Experimentation |

## License

MIT — see [LICENSE](LICENSE).
