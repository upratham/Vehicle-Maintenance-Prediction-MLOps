# 🚗 Vehicle Maintenance Prediction
### An End-to-End MLOps System (MSML605 Final Project)

---

**Live Demo (Render):** https://vehicle-maintenance-prediction-mlops.onrender.com

A production-grade MLOps system that predicts near-term vehicle maintenance needs. It ships with **live drift monitoring, one-click retraining, experiment tracking, and SHAP-grounded prediction explanations** surfaced directly in the product UI — not just a classifier wrapped in a form.

> 🎯 **Goal:** Predict whether a vehicle requires maintenance, and make the entire ML lifecycle (train → evaluate → promote → serve → monitor → retrain) visible and operable from a web dashboard.

See **[PRESENTATION.md](PRESENTATION.md)** for the full project deep-dive and demo script.

---

## ⚡ Quick Start (get it running in ~5 minutes)

### Prerequisites
- **Python 3.10+** (virtualenv/conda)
- **Node 20+** and **npm** (for the React frontend)
- **MongoDB Atlas** connection string (or any reachable Mongo URI with the project data loaded)
- **AWS** credentials with access to the S3 model bucket
- **ClearML** account (free tier at [app.clear.ml](https://app.clear.ml)) — *optional for serving, required for training*

### 1. Clone and enter the repo
```bash
git clone https://github.com/upratham/Vehicle-Maintenance-Prediction-MLOps.git
cd Vehicle-Maintenance-Prediction-MLOps
```

### 2. Python environment + dependencies
```bash
# Option A — venv (recommended for local dev)
python3.10 -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS (zsh/bash)
source venv/bin/activate

# Linux (bash)
source venv/bin/activate

pip install -r requirements.txt

# Option B — conda
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

### 3. Frontend dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. Environment variables

Create a `.env` file in the project root:

```bash
# ── MongoDB (required for training & drift logging) ──
CONNECTION_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/
DB_USERNAME=vehicle_db
COLLECTION_NAME=vehicles

# ── AWS S3 (required to load the deployed model) ──
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# ── ClearML experiment tracking (required for training) ──
CLEARML_API_HOST=https://api.clear.ml
CLEARML_API_ACCESS_KEY=your_clearml_access_key
CLEARML_API_SECRET_KEY=your_clearml_secret_key

# ── Retrain endpoint auth ──
OPS_TOKEN=any-long-random-string-you-pick

# ── Runtime ──
APP_ENV=dev
```

> 💡 Ask a team member (or read `PRESENTATION.md` § 14) for the actual credential values. Never commit `.env`.

### 5. Start the servers

**Easiest — use the dev scripts:**
```bash
bash .dev/dev.sh    # both backend + frontend in one terminal
# or separately:
bash .dev/be.sh     # FastAPI on http://localhost:5000
bash .dev/fe.sh     # Vite dev server on http://localhost:3000
```

Press `r` to restart both, `q` to quit. Pass `s` to run in staging mode: `bash .dev/dev.sh s`.

**Manual (if the dev scripts aren't your thing):**
```bash
# Terminal 1 — backend

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS (zsh/bash)
source venv/bin/activate

# Linux (bash)
source venv/bin/activate

python -m uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# Terminal 2 — frontend
cd frontend
npm run dev
```

### 6. Open it up
- **Main predictor:** http://localhost:3000
- **Ops dashboard:** http://localhost:3000/ops (model registry, baselines, drift chart, retrain)
- **Backend API docs:** http://localhost:5000/docs (FastAPI Swagger UI)

The Vite dev server proxies `/predict`, `/model_info`, `/drift`, and `/train` to the backend automatically — no CORS setup needed in dev.

---

## 🏋️ Running a training pipeline locally

The one-click retrain button on `/ops` is the recommended path, but you can also trigger training from the CLI:

```bash
python demo.py
```

This runs `TrainPipeline.run_pipeline()` end-to-end:
1. Ingest from MongoDB
2. Validate against `config/schema.yaml`
3. Transform + SMOTE oversample
4. Train the Keras ANN + sklearn baselines (LR, RF, XGBoost)
5. Log everything to ClearML (check [app.clear.ml](https://app.clear.ml))
6. Evaluate candidate vs. current deployed model
7. Promote to S3 if F1 beats the threshold

Artifacts land in `artifact/<timestamp>/` and the preprocessor gets saved to `preprocessor_obj/preprocessing.pkl`.

---

## 🏗️ System architecture

```
MongoDB Atlas                                             AWS S3
    │                                                       ▲
    ▼                                                       │
Data Ingestion ─► Validation ─► Transformation ─► Trainer ─►┤  Model registry
(src/components)   (YAML schema)  (RobustScaler,    (Keras  │  + baselines +
                                   OHE, ordinal,     ANN +  │  distribution
                                   SMOTE)            sklearn│  snapshot
                                                    baselines)
                                                            │
ClearML (experiment tracking) ◄─────┐                       │
                                    │                       │
                                    ▼                       │
                            FastAPI (app.py) ───────────────┘
                            /predict  /model_info
                            /train    /drift
                                    │
                                    ▼
                           React UI (frontend/)
                           • Main: VIN decode + SHAP-explained prediction
                           • /ops: registry, drift, retrain console
                                    │
                                    ▼
                           Docker → ECR → EC2 (GitHub Actions)
```

---

## 🌐 API endpoints

| Route | Method | Description | Auth |
|---|---|---|---|
| `/` | GET | Legacy HTML form (still works) | — |
| `/predict` | POST | JSON prediction + SHAP explanations | — |
| `/model_info` | GET | Current model version, hash, metrics, baselines | — |
| `/drift?window=7d` | GET | Per-feature PSI vs. training distribution | — |
| `/train` | POST | Streams training logs via SSE; returns old-vs-new metrics | `X-Ops-Token` header |
| `/docs` | GET | FastAPI Swagger UI | — |

---

## 🛠️ Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| **Language** | Python 3.10, TypeScript 5 | Backend + frontend |
| **Backend** | FastAPI + Uvicorn | JSON API + SSE |
| **Frontend** | React 19 + Vite + Tailwind | `/` predictor, `/ops` dashboard |
| **Data store** | MongoDB Atlas | Raw training data + prediction logs |
| **Model registry** | AWS S3 | Versioned `model.pkl` + manifest |
| **Experiment tracking** | ClearML (hosted) | Hyperparams, scalars, confusion matrix |
| **Explainability** | SHAP | Per-prediction feature attribution |
| **Training** | Keras (TF), scikit-learn, imblearn (SMOTE) | ANN + baselines |
| **Container** | Docker | Reproducible runtime |
| **Compute** | AWS EC2 + ECR | Self-hosted runner, image registry |
| **CI/CD** | GitHub Actions | Build → push → pull → restart |
| **Drift stat** | PSI (Population Stability Index) | Marginal distribution shift |

---

## 🗂️ Project structure

```
Vehicle-Maintenance-Prediction-MLOps/
│
├── app.py                        # FastAPI entry point
├── demo.py                       # CLI training trigger
├── requirements.txt              # Python deps
├── Dockerfile                    # Container spec
├── PRESENTATION.md               # Full project overview + demo script
│
├── src/
│   ├── components/               # Pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py      # ANN + baselines + ClearML
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py       # Writes model_registry.json
│   ├── pipline/                  # Orchestrators (typo preserved)
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py # Serves /predict + SHAP
│   ├── drift.py                  # PSI computation
│   ├── insights.py               # Baseline model training utilities
│   ├── entity/                   # Config + artifact dataclasses
│   ├── configuration/            # Mongo + AWS connection helpers
│   ├── cloud_storage/            # S3 wrapper
│   ├── data_access/
│   ├── utils/
│   └── constants/
│
├── frontend/                     # React 19 + Vite app
│   ├── src/
│   │   ├── App.tsx               # Main predictor
│   │   ├── pages/Ops.tsx         # MLOps dashboard
│   │   ├── components/           # VinPanel, ConditionPanel, ResultCard (SHAP bars)
│   │   └── lib/                  # NHTSA, mapping, ops API helpers
│   ├── package.json
│   └── vite.config.ts            # Proxies /predict etc. to :5000 in dev
│
├── config/
│   └── schema.yaml               # Data validation contract
├── notebooks/                    # EDA + feature engineering
├── plots/                        # Training visualizations
├── preprocessor_obj/             # Pickled preprocessor (post-training)
├── data/                         # CSV inputs
├── static/                       # Legacy HTML assets
├── templates/                    # Legacy Jinja templates
│
├── .dev/                         # Dev server scripts (dev.sh, fe.sh, be.sh)
├── .env                          # Local secrets (gitignored)
├── .env.staging                  # Staging overrides (gitignored)
└── .github/workflows/aws.yaml    # CI/CD pipeline
```

---

## 🔄 ML pipeline (what each stage does)

1. **Data Ingestion** — pulls from MongoDB Atlas, splits into train/test CSVs.
2. **Data Validation** — checks column names, types, and ranges against `config/schema.yaml`. Fails the pipeline early on schema drift.
3. **Data Transformation** — drops low-signal features, caps outliers (IQR), ordinal-encodes condition fields, one-hot-encodes nominal fields, `RobustScaler` on numerics, SMOTE on the minority class. Also emits `training_distribution.json` for drift comparison.
4. **Model Trainer** — trains a Keras ANN (primary) and sklearn baselines (Logistic Regression, Random Forest, XGBoost). Logs hyperparameters, scalars, and the confusion matrix to ClearML. Persists `baselines.json`.
5. **Model Evaluation** — compares candidate F1 vs. the currently deployed model on S3. Promotion requires an F1 improvement of at least `0.02`.
6. **Model Pusher** — uploads the winning model to S3 *and* writes `model_registry.json` (version, SHA-256, timestamp, metrics, S3 URI).

---

## 🚀 CI/CD pipeline

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

Required GitHub repo secrets:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS programmatic access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_DEFAULT_REGION` | Target AWS region (`us-east-1`) |
| `ECR_REPO` | ECR repository URI |

---

## 🧪 Smoke test (before demo / submission)

```bash
bash .dev/dev.sh       # start both servers
```

1. Open http://localhost:3000, run a prediction with VIN `1HGCM82633A123456`. **Expect:** verdict + SHAP bar chart.
2. Open http://localhost:3000/ops. **Expect:** model registry card, baselines table, drift chart.
3. Click **Retrain** (paste your `OPS_TOKEN`). **Expect:** streaming logs, old-vs-new metric diff, new ClearML task at [app.clear.ml](https://app.clear.ml).
4. Fire ~20 predictions with varied inputs. Refresh `/ops`. **Expect:** drift chart with per-feature PSI.

---

## 🧰 Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Backend didn't start` from `be.sh` | `venv` missing or `requirements.txt` not installed. Re-run step 2. |
| `npm run dev` fails with TS errors | Run `npm install` inside `frontend/` first. |
| `/predict` returns 500 | Model not on S3 yet, or AWS creds missing. Run `python demo.py` once or check `.env`. |
| `/train` returns 401 | `X-Ops-Token` header doesn't match `OPS_TOKEN` in `.env`. |
| ClearML complains about credentials | Run `clearml-init` once to set up `~/clearml.conf`, or set the three `CLEARML_*` env vars. |
| Frontend can't reach backend | Confirm `vite.config.ts` proxy points at `http://localhost:5000` and backend is actually up. |

---

## 👥 Authors

| Name | Role |
|---|---|
| **Prathamesh Uravane** | Deployment & MLOps lead |
| **Sankeerth B** | Data & Validation lead |
| **Claude B** | Modeling & Experimentation lead |

Full contribution breakdown in [PRESENTATION.md § 12](PRESENTATION.md).

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

⭐ *If you found this project helpful or interesting, please consider giving it a star!*
