# Vehicle Maintenance Prediction — Project Report
## MSML 605: Computing Systems for ML

**Authors:** Prathamesh Uravane | Vedant Ganthade
**Date:** April 2026

---

## 1. Project Overview

This project builds an end-to-end MLOps platform for vehicle maintenance prediction. The system goes beyond training a model — it deploys, monitors, versions, and retrains models in production through an automated pipeline and an operator-facing console.

The platform serves predictions for three distinct vehicle datasets through a unified API, backed by MongoDB for data and artifact storage, AWS S3 for model storage, and a React frontend that doubles as an MLOps operations console.

---

## 2. Problem Statement

Unplanned vehicle failures are costly, dangerous, and preventable. Traditional maintenance schedules are time-based rather than condition-based, resulting in either over-maintenance (wasted cost) or under-maintenance (unexpected failures).

Machine learning can bridge this gap — but only if the models can be reliably deployed, monitored for degradation, and retrained when data distributions shift. This project addresses both the prediction problem and the operational problem of running ML at scale.

---

## 3. Datasets

### 3.1 Vehicle Maintenance Dataset (Primary)
- **Size:** 50,000 records
- **Source:** MongoDB Atlas (`vehicle_maintenance_data` collection)
- **Features:** Vehicle model, mileage, maintenance history, reported issues, vehicle age, fuel type, transmission, engine size, odometer, service history, accident history, fuel efficiency, tire/brake/battery condition
- **Target:** `Need_Maintenance` (binary: 0 or 1)
- **Challenge:** Class imbalance (80% positive class) handled with SMOTE

### 3.2 Hyundai Cars Dataset
- **Size:** 1,100 records
- **Source:** MongoDB Atlas (`cars_hyundai` collection)
- **Features:** Engine temperature, brake pad thickness, tire pressure
- **Target:** `Maintenance Type` (3-class: Repair, Routine Maintenance, Component Replacement) + `Anomaly Indication` (binary)
- **Challenge:** Small dataset requiring careful regularization

### 3.3 Engine Condition Dataset
- **Size:** 19,535 records
- **Source:** MongoDB Atlas (`engine_data` collection)
- **Features:** Engine RPM, lubricating oil pressure/temperature, fuel pressure, coolant pressure/temperature
- **Target:** `Engine Condition` (binary: Healthy/Faulty)
- **Challenge:** Engineered 14 domain-specific features (temp ratios, RPM-stress interactions, risk flags)

---

## 4. ML Pipeline Architecture

### 4.1 Pipeline Stages

Every training profile passes through identical pipeline stages:

```
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation → Model Pusher
```

1. **Data Ingestion:** Pulls data from MongoDB, splits into train/test (75/25 stratified)
2. **Data Validation:** Checks schema, column types, value ranges against YAML schema files
3. **Data Transformation:** Ordinal encoding, one-hot encoding, robust scaling, SMOTE for class imbalance, mutual information feature selection
4. **Model Training:** HPO via Optuna (20 trials, 3-fold CV), logged to ClearML
5. **Model Evaluation:** Compares new model against production model on F1; accepts if improvement > 2%
6. **Model Pusher:** Uploads accepted model to AWS S3, writes manifest to MongoDB artifact store

### 4.2 Pipeline Profiles (YAML-Driven)

The pipeline is driven by `config/pipeline_profiles.yaml`:

```yaml
profiles:
  vehicle_maintenance:
    dataset:
      collection: vehicle_maintenance_data
      target_column: NEED_MAINTENANCE
    model:
      type: rf_hpo
  cars_hyundai:
    dataset:
      collection: cars_hyundai
      target_column: Maintenance Type
    model:
      type: svm_rf_hpo
  engine_data:
    dataset:
      collection: engine_data
      target_column: Engine Condition
    model:
      type: rf_hpo
```

Adding a new dataset requires only a new YAML entry — no Python code changes.

### 4.3 Model Types

| Profile | Algorithm | HPO |
|---|---|---|
| vehicle_maintenance | Random Forest | Optuna (n_estimators, max_depth, min_samples) |
| cars_hyundai | SVM + Random Forest ensemble | Grid search |
| engine_data | Keras ANN (2-layer, BatchNorm, Dropout) | Manual |

### 4.4 Baseline Comparison

After every training run, four baseline models are automatically trained and compared:
- Logistic Regression
- Random Forest (fixed params)
- SVM (RBF kernel)
- XGBoost

Results are stored in MongoDB and displayed in the MLOps console.

---

## 5. Novel Contributions

### 5.1 Multi-Model Orchestrator (Profile-Based Routing)

**What it is:** A single prediction API routes requests to the correct model based on a `profile_name` parameter, with each model loading its own preprocessor and S3 model independently.

**Why it is novel:** Most tutorials deploy a single model. Our orchestrator handles N models with different feature spaces, preprocessing pipelines, and model types — all behind one unified interface.

```python
class MultiModelOrchestrator:
    CLASSIFIERS = {
        "vehicle_maintenance": VehicleMaintenanceClassifier,
        "cars_hyundai":        HyundaiCarsClassifier,
        "engine_data":         EngineDataClassifier,
    }
    def predict(self, profile_name: str, dataframe: DataFrame) -> dict:
        classifier = self.CLASSIFIERS[profile_name]()
        raw = classifier.predict(dataframe)[0]
        score = float(raw)
        label = 1 if score >= 0.5 else 0
        return {"profile": profile_name, "label": label, "score": score, "status": classifier.label(label)}
```

**Extensibility:** To add a fourth model, add one class and one YAML entry. Zero API changes.

---

### 5.2 Dual Docker Image Orchestration

**Architecture:**

```
ECR Repository (vehicleproj)
├── :backend   — python:3.12-slim + FastAPI + ML stack  (~690 MB)
└── :frontend  — nginx:alpine + React build             (~26 MB)
```

**Orchestration via docker-compose.prod.yml on EC2:**

```yaml
services:
  backend:
    image: ${REGISTRY}/${REPO}:backend
    ports: ["8000:8000"]
    environment:
      CONNECTION_URL: ...
      AWS_ACCESS_KEY_ID: ...

  frontend:
    image: ${REGISTRY}/${REPO}:frontend
    ports: ["80:80"]
    depends_on: [backend]
```

**Frontend Dockerfile (multi-stage build):**
- Stage 1: Node.js builds the React/Vite app (`npm run build`)
- Stage 2: nginx serves the static `dist/` folder and proxies `/api/*` → backend:8000

**Why two images?**
1. **Independent deployment:** A CSS fix deploys only the 26 MB frontend image, not the 690 MB ML backend
2. **Security isolation:** The frontend container has no access to AWS credentials or ML code
3. **Nginx reverse proxy:** The frontend container handles CORS, SSL termination, and API proxying — the backend only speaks FastAPI

**How they communicate:**
- Browser → nginx (port 80) → React app served statically
- React API calls → nginx proxies to backend (port 8000) internally on Docker network
- No direct browser-to-backend communication needed

---

### 5.3 MongoDB-Backed Artifact Store

**Problem:** Docker containers are ephemeral. The `artifact/` directory (containing model registry, baselines, training distributions) is lost every time containers are restarted or EC2 is rebooted.

**Solution:** A dedicated `artifact_store.py` module that persists training artifacts to MongoDB after every run.

**Collections:**

| Collection | Contents | Retention |
|---|---|---|
| `artifact_model_registry` | Model version, metrics, S3 URI, SHA256 | Latest 5 per profile |
| `artifact_baselines` | LR/RF/SVM/XGB comparison table | Latest 5 per profile |
| `prediction_logs` | Every live prediction (input + score + label) | Unbounded (TTL index recommended) |
| `repair_costs` | Service cost lookup table | Static |

**Auto-pruning:** After each insert, documents older than the 5 most recent are deleted:

```python
ids = col.find({"profile_name": profile_name}, {"_id": 1}).sort("created_at", -1).skip(5)
col.delete_many({"_id": {"$in": ids}})
```

**Read strategy:** Every function that reads an artifact tries MongoDB first and falls back to the local filesystem — so the system works even when MongoDB is temporarily unreachable.

---

### 5.4 Population Stability Index (PSI) Drift Monitoring

**What it measures:** How much the live prediction input distribution has shifted from the training distribution.

**Formula:**
```
PSI = Σ (live_pct - train_pct) × ln(live_pct / train_pct)
```

**Thresholds:**

| PSI | Interpretation |
|---|---|
| < 0.10 | Stable — no action needed |
| 0.10 – 0.25 | Warning — monitor closely |
| > 0.25 | Drifted — retrain recommended |

**Implementation:**
1. At transformation time, numerical feature histograms and categorical frequency tables are saved to `training_distribution.json`
2. Every live prediction is logged to MongoDB `prediction_logs` with timestamp
3. On `/drift?window=7d`, PSI is computed per feature comparing the live 7-day window against training histograms
4. Results are displayed per feature in the MLOps console with colour-coded verdicts

---

### 5.5 Live Training Log Streaming (SSE)

When the operator clicks **Retrain** in the MLOps console, the backend:
1. Spawns a worker thread running the full training pipeline
2. All Python `logging` calls are forwarded to a thread-safe queue
3. The HTTP response is a `text/event-stream` (Server-Sent Events)
4. Each log line is streamed to the browser in real-time
5. At completion, a JSON summary event carries the before/after metric diff

This means training progress is visible live in the browser without polling.

---

### 5.6 Rule-Based Service Recommendation

After every prediction, the backend returns not just a binary label but a **service recommendation** with cost estimate:

```json
{
  "status": "Maintenance Required",
  "score": 0.87,
  "service": {
    "name": "Brake pad replacement",
    "cost_low": 180,
    "cost_high": 450,
    "hours_low": 1.0,
    "hours_high": 2.5,
    "notes": "Front or rear pads; includes labor"
  },
  "impacts": [...]
}
```

Service costs are loaded from MongoDB `repair_costs` collection with hardcoded fallback — so predictions never fail due to a missing CSV file.

---

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Main vehicle maintenance prediction (form + JSON) |
| POST | `/predict/hyundai` | Hyundai cars anomaly detection |
| POST | `/predict/engine` | Engine condition prediction |
| GET | `/model_info` | Current model registry + baselines |
| GET | `/drift?window=7d` | Per-feature PSI drift report |
| POST | `/train` | Trigger retraining (SSE stream) |

---

## 7. CI/CD Pipeline

```
git push → GitHub Actions
    │
    ├── Continuous-Integration (GitHub Cloud)
    │   ├── docker build backend → push to ECR
    │   └── docker build frontend → push to ECR
    │
    └── Continuous-Deployment (self-hosted EC2 runner)
        ├── Fix workspace permissions
        ├── docker system prune (free disk)
        ├── docker pull :backend + :frontend from ECR
        ├── docker-compose down
        └── docker-compose up -d
```

**Self-hosted runner:** The EC2 instance runs a GitHub Actions runner as a systemd service (`actions.runner.*.service`), which connects to GitHub and executes CD jobs locally. This gives the pipeline direct access to EC2 resources (Docker daemon, environment variables) without exposing the instance publicly.

---

## 8. Infrastructure Summary

| Component | Service |
|---|---|
| Compute | AWS EC2 (t3.small, 20GB EBS) |
| Container Registry | AWS ECR (`vehicleproj` repo) |
| Model Storage | AWS S3 (`vehicle-maintenance-prediction-model`) |
| Database | MongoDB Atlas (free tier) |
| Experiment Tracking | ClearML (app.clear.ml) |
| CI/CD | GitHub Actions |
| DNS / Networking | EC2 public IP + Security Group (ports 80, 8000, 22) |

---

## 9. Probable Professor Questions and Answers

---

**Q1: Why did you use three separate models instead of one unified model?**

Each dataset has a fundamentally different feature space and prediction target. The Vehicle Maintenance dataset has 20 features including categorical vehicle metadata. The Engine Condition dataset has 6 continuous sensor readings. The Hyundai dataset is a small 1,100-record dataset with only 3 features. Training a single model across these would require a shared feature schema — which is impossible without discarding most features. The multi-model orchestrator pattern lets each dataset have its own optimized preprocessor, feature space, and model type while presenting a single API surface.

---

**Q2: How does your system handle model drift?**

We implement Population Stability Index (PSI) drift monitoring. At training time, we save a histogram of each feature's distribution. Every live prediction is logged to MongoDB with its input values. The `/drift` endpoint computes PSI per feature by comparing the live distribution (over a configurable time window, e.g., 7 days) against the training histogram. PSI < 0.10 is stable; 0.10–0.25 is a warning; > 0.25 triggers a retrain recommendation. The results are displayed per-feature in the MLOps console.

---

**Q3: Why are you using MongoDB for model artifacts instead of just S3 or the filesystem?**

The artifact directory (`model_registry.json`, `baselines.json`) is on the EC2 filesystem, which is ephemeral — it's wiped on container restart or EC2 termination. S3 is possible but adds latency and complexity for small JSON documents. MongoDB Atlas is already in the infrastructure (used for training data and prediction logs), is always available, and is well-suited for small structured documents. We also wanted to keep a history of the last 5 training runs per profile — MongoDB's query capabilities make this trivial with a `sort().skip(N).delete()` pattern.

---

**Q4: What is the role of the nginx container?**

nginx serves two purposes. First, it serves the React static files (HTML, JS, CSS bundles) that were compiled during the Docker image build. Second, it acts as a reverse proxy — API calls from the React app go to `/api/...` on port 80, and nginx rewrites and forwards these to the FastAPI backend on port 8000 on the internal Docker network. This means the browser never needs to know the backend's port, and we avoid CORS issues entirely because from the browser's perspective, everything is on the same origin.

---

**Q5: How does your CI/CD pipeline ensure the EC2 instance always has enough disk space?**

We learned this lesson through failures. The Docker images (especially with TensorFlow) were several GB, and the EC2's 8 GB disk would fill up after a few deployments because old images weren't being cleaned. We added two fixes: (1) a `docker system prune -af` step runs before every pull to remove dangling and unused images, and (2) we resized the EBS volume from 8 GB to 20 GB. We also moved `torch` and `tensorflow` out of the production image and into notebook-level `!pip install` cells, reducing the backend image size by approximately 2.5 GB.

---

**Q6: What is the purpose of the self-hosted GitHub Actions runner on EC2?**

GitHub-hosted runners run on GitHub's cloud and have no access to your EC2 instance. For the deployment job, we need to run `docker pull` and `docker-compose up` directly on the EC2 instance. A self-hosted runner is a process on EC2 that registers with GitHub, receives job assignments, and executes them locally. This gives the CI/CD pipeline direct access to the EC2 Docker daemon and environment without requiring SSH or public API exposure. It runs as a systemd service so it restarts automatically on reboot.

---

**Q7: How does the retrain pipeline avoid deploying a worse model?**

The `ModelEvaluation` component compares the newly trained model against the currently deployed model (loaded from S3). It computes F1 score on the held-out test set for both models. The new model is only accepted and pushed to S3 if its F1 score is higher than the production model's score by more than a configured threshold (default 2%). If the new model is worse, the pipeline logs "Model not accepted" and the existing S3 model continues to serve predictions unchanged.

---

**Q8: Why did you choose FastAPI over Flask or Django?**

FastAPI offers automatic OpenAPI/Swagger documentation, native async support (important for the SSE training log stream), Pydantic-based request validation with automatic error responses, and significantly higher throughput than Flask. The SSE endpoint (`/train`) requires async generators which FastAPI handles natively. Flask would require additional libraries (flask-sse) and a message broker for comparable functionality.

---

**Q9: What does the `preprocessing.pkl` file contain and why is it baked into the Docker image?**

The preprocessor is a scikit-learn `ColumnTransformer` fitted on the training data. It contains the learned parameters for ordinal encoding (category orders), one-hot encoding (learned category lists), and robust scaling (learned median and IQR per feature). It must be identical between training and inference — using a different or re-fitted preprocessor would produce different feature representations and incorrect predictions. It is baked into the Docker image (via `COPY preprocessor_obj /app/preprocessor_obj`) because it must be available immediately on container start without requiring a training run.

---

**Q10: How would you scale this system to handle 10x more traffic?**

Currently the system runs on a single EC2 instance. To scale: (1) Put an Application Load Balancer (ALB) in front of multiple EC2 instances running the backend container. (2) Use Amazon ECS or Kubernetes to manage container scaling automatically based on CPU/memory metrics. (3) The stateless design (models loaded from S3, data from MongoDB Atlas) means any instance can handle any request — horizontal scaling is straightforward. (4) MongoDB Atlas can be upgraded to a larger tier. (5) For the frontend, serve from CloudFront CDN instead of nginx on EC2.

---

**Q11: What is ClearML and why did you use it alongside your own MLOps console?**

ClearML is a full-featured experiment tracking platform that records every training run including hyperparameter values, per-epoch loss/accuracy curves, confusion matrices, and model artifacts. We use it for deep experiment analysis and reproducibility — you can click any experiment in ClearML and re-run it exactly. Our custom MLOps console in the React frontend serves a different purpose: it's the operational view for non-data-scientists — showing the current deployed model's metrics, comparing it to baselines, and triggering retrains. ClearML is for data scientists; the console is for operators.

---

**Q12: Explain the Population Stability Index formula.**

PSI measures distribution shift by comparing expected (training) and actual (live) distributions:

```
PSI = Σ_bins (Actual% - Expected%) × ln(Actual% / Expected%)
```

For each bin/category, we compute the fraction of training samples and live samples that fall in it. The difference weighted by the log ratio gives a scalar per bin; summing across all bins gives PSI for that feature. The formula is asymmetric — it penalizes large shifts in both directions. A PSI of 0 means identical distributions. Values above 0.25 indicate a significant shift that would likely degrade model performance, triggering a retrain recommendation.

---

*End of Report*
