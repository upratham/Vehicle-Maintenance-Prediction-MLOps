# Vehicle Maintenance Prediction — MLOps System
### MSML 605 Final Project · End-to-End Multi-Model Pipeline

**Live Demo (Render):** https://vehicle-maintenance-prediction-mlops.onrender.com

A production-grade MLOps system that trains, evaluates, and serves three independent vehicle-health classifiers — one per dataset — through a single unified pipeline. Each model gets its own dataset-specific transformer class, its own ClearML-logged Optuna HPO run, and its own S3 model registry entry.

> **Goal:** Cover the full ML lifecycle (ingest → validate → transform → HPO train → evaluate → promote → serve → monitor → retrain) across three datasets, with every experiment tracked in ClearML and every model promoted via S3.

See **[PRESENTATION.md](PRESENTATION.md)** for the full project deep-dive and demo script.

---

## Three datasets, three models

| Profile | Dataset | Rows | Target | Model | HPO |
|---|---|---|---|---|---|
| `vehicle_maintenance` | `vehicle_maintenance_data.csv` | 50 000 | `Need_Maintenance` (binary) | Random Forest | Optuna (20 trials, 3-fold CV F1) |
| `cars_hyundai` | `cars_hyundai.csv` | 1 100 | `Anomaly Indication` (binary) | Decision Tree | Optuna (20 trials, 3-fold CV F1) |
| `engine_data` | `engine_data.csv` | 19 535 | `Engine Condition` (binary) | Random Forest | Optuna (20 trials, 3-fold CV F1) |

Each profile runs the same six-stage pipeline sequentially. All HPO trial metrics stream to ClearML in real time.

---

## Quick Start (≈ 5 minutes)

### Prerequisites
- **Python 3.12** (required — see `pyproject.toml`)
- **Node 20+** and **npm** (React frontend)
- **MongoDB Atlas** URI (or any reachable Mongo with the project data loaded)
- **AWS** credentials with read/write access to the S3 model bucket
- **ClearML** account (free at [app.clear.ml](https://app.clear.ml)) — required for training, optional for serving

### 1. Clone
```bash
git clone https://github.com/upratham/Vehicle-Maintenance-Prediction-MLOps.git
cd Vehicle-Maintenance-Prediction-MLOps
```

### 2. Python environment
```bash
python3.12 -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend
```bash
cd frontend && npm install && cd ..
```

### 4. Environment variables

Create `.env` in the project root:

```bash
# MongoDB (required for training)
CONNECTION_URL=mongodb+srv://<user>:<pass>@cluster.mongodb.net/
DB_USERNAME=605_Project_Data
COLLECTION_NAME=vehicle_maintenance_data

# AWS S3 (required to load / push models)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# ClearML (required for training)
CLEARML_API_HOST=https://api.clear.ml
CLEARML_API_ACCESS_KEY=your_clearml_access_key
CLEARML_API_SECRET_KEY=your_clearml_secret_key

# Retrain endpoint auth
OPS_TOKEN=any-long-random-string

APP_ENV=dev
```

> Never commit `.env`. Ask a team member for the actual values, or see `PRESENTATION.md § 14`.

### 5. Start servers
```bash
bash .dev/dev.sh          # backend + frontend together
# or separately:
bash .dev/be.sh           # FastAPI on http://localhost:5000
bash .dev/fe.sh           # Vite on http://localhost:3000
```

### 6. Open
| URL | What you get |
|---|---|
| http://localhost:3000 | Predictor (VIN decode + SHAP explanation) |
| http://localhost:3000/ops | Ops dashboard (registry, baselines, drift, retrain) |
| http://localhost:5000/docs | FastAPI Swagger UI |

---

## Running the training pipeline

```bash
python demo.py
```

Runs `TrainPipeline.run_pipeline()` across all three profiles in sequence. Local CSVs in `data/` are used automatically — MongoDB is only hit if a matching CSV is not found, avoiding slow cloud round-trips.

To run a single profile:

```python
from src.pipline.training_pipeline import TrainPipeline
TrainPipeline().run_pipeline(profile_names=["engine_data"])
```

To force a fresh MongoDB fetch and re-sync:

```python
TrainPipeline().run_pipeline(refresh_collections=True)
```

Artifacts land in `artifact/<profile>/<timestamp>/`. The preprocessor is saved to `preprocessor_obj/<profile>/preprocessing.pkl`.

---

## Pipeline stages (per profile)

```
data/                    MongoDB Atlas (fallback)
  └─ <collection>.csv
         │
         ▼
 1. Data Ingestion       Loads local CSV (or fetches from MongoDB).
         │               Writes to artifact/<profile>/<ts>/data_ingestion/
         ▼
 2. Data Validation      Checks column names + types against config/schema_<profile>.yaml.
         │               Fails early on schema drift.
         ▼
 3. Data Transformation  Profile-specific class (see below).
         │               Emits training_distribution.json for drift comparison.
         ▼
 4. Model Trainer HPO    Profile-specific class runs Optuna HPO.
         │               Every trial F1 is streamed to ClearML (hpo/cv_f1).
         │               Best params stored in ClearML task + baselines.json.
         ▼
 5. Model Evaluation     Compares candidate F1 vs. deployed model on S3.
         │               Requires F1 delta >= 0.02 to promote.
         ▼
 6. Model Pusher         Uploads winning model to S3.
                         Writes model_registry.json (version, SHA-256, metrics).
```

### Transformation classes

| Class | Profile | Key steps |
|---|---|---|
| `VehicleMaintenanceDataTransformation` | `vehicle_maintenance` | Drop low-signal features → MI filter → outlier capping (IQR) → OrdinalEncoder + OHE + RobustScaler → SMOTE |
| `HyundaiCarsDataTransformation` | `cars_hyundai` | OHE + RobustScaler (no SMOTE — dataset is balanced) |
| `EngineDataTransformation` | `engine_data` | RobustScaler only (all-numeric dataset) → SMOTE |
| `DataTransformation` | dispatcher | Selects the right class by `profile_name` (same `__new__` pattern as model trainer) |

### Trainer classes

| Class | Profile | Model | Hyperparameter search space |
|---|---|---|---|
| `VehicleMaintenanceModelTrainer` | `vehicle_maintenance` | `RandomForestClassifier` | `n_estimators` [100–600], `max_depth` [4–30], `min_samples_leaf` [1–10], `min_samples_split` [2–20], `max_features` {sqrt, log2} |
| `HyundaiCarsModelTrainer` | `cars_hyundai` | `DecisionTreeClassifier` | `max_depth` [3–25], `min_samples_leaf` [1–20], `min_samples_split` [2–30], `criterion` {gini, entropy}, `max_features` {sqrt, log2, all} |
| `EngineDataModelTrainer` | `engine_data` | `RandomForestClassifier` | `n_estimators` [100–600], `max_depth` [3–25], `min_samples_leaf` [1–10], `min_samples_split` [2–20], `max_features` {sqrt, log2} |
| `ModelTrainer` | dispatcher | — | Selects trainer by `profile_name` via `__new__` |

HPO trial count and CV folds are configurable per profile via `params.hpo_trials` and `params.hpo_cv_folds` in `src/constants/__init__.py` or the profile's model YAML.

---

## System architecture

```
data/<collection>.csv  ──┐
MongoDB Atlas  ──────────┤ (fallback if CSV missing)
                         │
                         ▼
                  Data Ingestion
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      vehicle_     cars_hyundai   engine_data
      maintenance  (HyundaiCars   (EngineData
      (VehicleMaint Transform)    Transform)
       Transform)
           │             │             │
           └─────────────┼─────────────┘
                         ▼
               Profile-specific HPO Trainer
              (Optuna trials → ClearML logging)
                         │
                         ▼
                  Model Evaluation
               (F1 vs. S3 production model)
                         │
                    (if accepted)
                         ▼
               Model Pusher → AWS S3
               model_registry.json
                         │
                         ▼
            ClearML  ←── FastAPI (app.py)
          (experiment        │
           tracking)    /predict  /model_info
                        /train    /drift
                             │
                             ▼
                     React UI (frontend/)
                   • / : predictor + SHAP bars
                   • /ops : registry, drift, retrain
                             │
                             ▼
                  Docker → ECR → EC2 (GitHub Actions)
```

---

## API endpoints

| Route | Method | Description | Auth |
|---|---|---|---|
| `/` | GET | Legacy HTML form | — |
| `/predict` | POST | JSON prediction + SHAP explanations | — |
| `/model_info` | GET | Version, hash, metrics, baselines for all profiles | — |
| `/drift?window=7d` | GET | Per-feature PSI vs. training distribution | — |
| `/train` | POST | SSE-streamed training logs; returns old-vs-new metric diff | `X-Ops-Token` |
| `/docs` | GET | FastAPI Swagger UI | — |

---

## Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.12, TypeScript 5 | Backend + frontend |
| Backend | FastAPI + Uvicorn | JSON API + SSE |
| Frontend | React 19 + Vite + Tailwind | Predictor UI + Ops dashboard |
| Data store | MongoDB Atlas | Raw training data + prediction logs |
| Local cache | `data/<collection>.csv` | Avoids MongoDB round-trip on every run |
| Model registry | AWS S3 | Versioned `model.pkl` + `model_registry.json` per profile |
| Experiment tracking | ClearML | HPO curves, test metrics, best params, model artifact |
| HPO | Optuna | In-process hyperparameter search (20 trials, 3-fold CV) |
| Explainability | SHAP | Per-prediction feature attribution |
| ML | scikit-learn, imbalanced-learn (SMOTE), XGBoost | Classifiers + baselines |
| Deep learning | TensorFlow / Keras | Production model evaluation (legacy ANN models on S3) |
| Container | Docker | Reproducible runtime |
| Compute | AWS EC2 + ECR | Self-hosted runner, image registry |
| CI/CD | GitHub Actions | Build → push → pull → restart |
| Drift detection | PSI (Population Stability Index) | Marginal distribution shift |

---

## Project structure

```
Vehicle-Maintenance-Prediction-MLOps/
│
├── app.py                         # FastAPI entry point
├── demo.py                        # CLI pipeline trigger (all 3 profiles)
├── requirements.txt               # Python deps (pinned minimums)
├── pyproject.toml                 # Package metadata + dep constraints
├── Dockerfile
│
├── config/
│   ├── pipeline_profiles.yaml     # Profile definitions (dataset, model type, schema path)
│   ├── schema.yaml                # vehicle_maintenance validation schema
│   ├── schema_cars_hyundai.yaml   # cars_hyundai validation schema
│   ├── schema_engine_data.yaml    # engine_data validation schema
│   ├── model.yaml                 # vehicle_maintenance model config
│   ├── model_cars_hyundai.yaml    # cars_hyundai model config
│   └── model_engine_data.yaml     # engine_data model config
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # CSV-first load (MongoDB fallback)
│   │   ├── data_validation.py     # Schema validation
│   │   ├── data_transformation.py # 3 transformer classes + DataTransformation dispatcher
│   │   ├── model_trainer.py       # 3 HPO trainer classes + ModelTrainer dispatcher
│   │   ├── model_evaluation.py    # F1 comparison vs. S3 production model
│   │   └── model_pusher.py        # S3 upload + model_registry.json
│   ├── pipline/
│   │   ├── training_pipeline.py   # TrainPipeline.run_pipeline() — multi-profile orchestrator
│   │   └── prediction_pipeline.py # /predict + SHAP
│   ├── drift.py                   # PSI computation
│   ├── insights.py                # Feature impact + service recommendations
│   ├── entity/
│   │   ├── config_entity.py       # Config dataclasses (TrainingPipelineConfig etc.)
│   │   ├── artifact_entity.py     # Artifact dataclasses
│   │   ├── estimator.py           # Model wrapper
│   │   └── s3_estimator.py        # S3 model loader
│   ├── constants/__init__.py      # All constants + default HPO params
│   ├── configuration/             # MongoDB + AWS connection helpers
│   ├── cloud_storage/             # S3 wrapper
│   ├── data_access/               # MongoDB → DataFrame
│   └── utils/main_utils.py        # load/save object, numpy, YAML helpers
│
├── data/                          # Raw CSVs (used as local cache)
│   ├── vehicle_maintenance_data.csv
│   ├── cars_hyundai.csv
│   └── engine_data.csv
│
├── preprocessor_obj/              # Pickled ColumnTransformer per profile
│   ├── vehicle_maintenance/preprocessing.pkl
│   ├── cars_hyundai/preprocessing.pkl
│   └── engine_data/preprocessing.pkl
│
├── artifact/                      # Per-run artifacts (timestamped, gitignored)
│   ├── vehicle_maintenance/<ts>/
│   ├── cars_hyundai/<ts>/
│   └── engine_data/<ts>/
│
├── plots/                         # MI scores + Spearman heatmap (vehicle_maintenance)
├── frontend/                      # React 19 + Vite app
│   └── src/
│       ├── App.tsx                # Main predictor
│       ├── pages/Ops.tsx          # MLOps dashboard
│       └── components/            # VinPanel, ConditionPanel, ResultCard (SHAP bars)
│
├── .dev/                          # dev.sh, be.sh, fe.sh
└── .github/workflows/aws.yaml     # CI/CD: build → ECR → EC2
```

---

## ClearML experiment tracking

Each training run creates a separate ClearML task per profile:

- **Task type:** `optimizer` (reflects HPO nature)
- **Scalars logged:**
  - `hpo/cv_f1` — F1 for every Optuna trial (visible as a curve in ClearML UI)
  - `hpo/best_cv_f1` — final best cross-validated F1
  - `test/accuracy`, `test/f1`, `test/precision`, `test/recall`, `test/roc_auc` — held-out test set metrics
- **Parameters:** `profile_name`, `hpo_best_params` (all best hyperparameters)
- **Artifact:** `trained_model` (serialised sklearn model)

View runs at [app.clear.ml](https://app.clear.ml) under project `605-Vehicle_Maintainance-project`.

---

## CI/CD pipeline

```
Developer pushes to GitHub
        │
        ▼
GitHub Actions (.github/workflows/aws.yaml)
        │
        ▼
Build Docker image → Push to AWS ECR
        │
        ▼
Self-hosted EC2 runner pulls latest image
        │
        ▼
Container restarted → App live on EC2
```

Required GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `ECR_REPO`.

---

## Smoke test (before demo / submission)

```bash
bash .dev/dev.sh
```

1. Open http://localhost:3000, predict with VIN `1HGCM82633A123456`. Expect: verdict + SHAP bars.
2. Open http://localhost:3000/ops. Expect: model registry card, baselines table, drift chart.
3. Click **Retrain** (paste `OPS_TOKEN`). Expect: streamed logs, old-vs-new metric diff, new ClearML tasks.
4. Fire ~20 predictions with varied inputs. Refresh `/ops`. Expect: drift chart with per-feature PSI.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `demo.py` hangs at "Exporting data from mongodb" | Add local CSVs to `data/` — the pipeline uses them automatically and skips MongoDB. |
| `/predict` returns 500 | No model on S3 yet. Run `python demo.py` at least once. Check `.env` AWS credentials. |
| `/train` returns 401 | `X-Ops-Token` header doesn't match `OPS_TOKEN` in `.env`. |
| ClearML not logging | Run `clearml-init` once, or set `CLEARML_API_HOST/ACCESS_KEY/SECRET_KEY` in `.env`. |
| `npm run dev` TS errors | Run `npm install` inside `frontend/` first. |
| Unicode `→` in logs on Windows | Already fixed in source; delete `src/components/__pycache__` if the old `.pyc` persists. |

---

## Authors

| Name | Role |
|---|---|
| Prathamesh Uravane | Deployment & MLOps lead |
| Sankeerth B | Data & Validation lead |
| Claude B | Modeling & Experimentation lead |

Full contribution breakdown in [PRESENTATION.md § 12](PRESENTATION.md).

---

## License

MIT — see [LICENSE](LICENSE).
