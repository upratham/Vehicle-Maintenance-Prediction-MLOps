# Vehicle Maintenance Prediction — MLOps System

**MSML605 Final Project · Group:** Sankeerth B · Claude B · Prathamesh Uravane
**Live demo:** production FastAPI + React on AWS EC2 · **Repo:** this directory

This document is both the project overview and the speaker guide for our 10-minute live demo. Read it top-to-bottom; each section maps to one slide.

---

## 1. One-sentence pitch

> An end-to-end MLOps system that predicts near-term vehicle maintenance needs — with live drift monitoring, one-click retraining, experiment tracking, and SHAP-grounded prediction explanations surfaced directly in the product UI.

---

## 2. Problem statement

Fleet operators lose money to **unplanned breakdowns**. Reactive servicing is expensive; fixed-interval preventive maintenance ignores per-vehicle usage. We frame maintenance as a **binary classification** problem over vehicle attributes (age, mileage, reported issues, component condition) and ship a **containerized, cloud-deployed system** — not just a model — that bridges academic modeling and operable deployment.

---

## 3. What we actually built (the pipeline)

```
MongoDB Atlas                                             AWS S3
    │                                                       ▲
    ▼                                                       │
Data Ingestion ─► Validation ─► Transformation ─► Trainer ─►┤  Model registry
(src/components)   (YAML schema)  (RobustScaler,    (Keras  │  + baselines +
                                   OHE, ordinal,     ANN)   │  distribution
                                   SMOTE)                   │  snapshot
                                                            │
ClearML experiment tracking ◄───────┐                       │
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

**Key directories**
- `src/components/` — ingest, validate, transform, train, evaluate, push
- `src/pipline/` — training and prediction orchestrators
- `src/drift.py` — PSI computation helpers
- `app.py` — FastAPI endpoints + prediction-logging middleware
- `frontend/` — React + Vite, dark/light themed, `/ops` dashboard
- `config/schema.yaml` — data contract for validation
- `.github/workflows/aws.yaml` — CI/CD to ECR + EC2

---

## 4. Novelty (honest version)

We didn't invent an algorithm. The novelty is **making the MLOps lifecycle visible and interactive inside the product itself**:

1. **Prediction explainability baked into the response** — every `/predict` call returns SHAP contributions, rendered inline so users see *why* the model flagged their vehicle.
2. **Live drift monitoring** via MongoDB-logged predictions + PSI vs. the training distribution snapshot, surfaced as a per-feature gauge.
3. **One-click retraining** from the web UI with streaming logs, auto-promotion on metric improvement, and an audit trail in the model registry.
4. **Experiment tracking at training time** through ClearML — every run is versioned with hyperparams, scalars, confusion matrices, and artifacts.
5. **Baseline comparison exposed in the UI**, not buried in a notebook — Logistic Regression / Random Forest / XGBoost / ANN side-by-side with the winner highlighted.

That's what "bridging academic and industry" actually means in our implementation.

---

## 5. The six MLOps capabilities (demo script — one slide each)

### Slide A · Model Registry (`/ops` page)

- **Endpoint:** `GET /model_info` reads `model_registry.json` written on every successful push.
- **Fields shown:** version, trained-at timestamp, SHA-256 hash, S3 URI, current metrics (F1, precision, recall, ROC-AUC).
- **Speaker line:** "Every deployed model is pinned by hash. If an artifact on S3 changes, the UI shows it. That's the registry contract."

### Slide B · Baseline Comparison

- **Persisted:** `baselines.json` written during training with cross-validated metrics for LR, RF, XGBoost, and the ANN.
- **Rendered:** sortable table on `/ops`, winner highlighted.
- **Speaker line:** "Final report rubric demands baseline comparison. Here it is — not in a notebook appendix, but the same file the production system reads."

### Slide C · Prediction Explainability (SHAP)

- **Backend:** `VehicleDataClassifier.predict()` runs `shap.Explainer(model)` over the transformed input (background cached on first call), returns the top-5 contributions in the `/predict` response.
- **Frontend:** `ResultCard.tsx` renders a horizontal bar chart — ember bars push *toward* maintenance-needed, red bars push *away*.
- **Speaker line:** "The model's prediction is never a black box. Every output ships with feature-level attribution."

### Slide D · ClearML Experiment Tracking

- **Wiring:** `Task.init(project_name="605-Vehicle_Maintainance-project", ...)` inside `model_trainer.py`.
- **Logged:** hyperparameters, per-epoch scalars (accuracy, AUC, precision, recall), confusion matrix plot, final model artifact.
- **Speaker line:** Switch to the ClearML dashboard screenshot. "This is every training run we've ever done, versioned and comparable. Promoting a model means promoting a specific Task ID."

### Slide E · Drift Monitor

- **Logging:** FastAPI middleware writes every `/predict` payload + response to the `prediction_logs` Mongo collection with a timestamp.
- **Snapshot:** `data_transformation.py` emits `training_distribution.json` (histograms for numeric features, value counts for categoricals).
- **Endpoint:** `GET /drift?window=7d` computes Population Stability Index per feature using helpers in `src/drift.py`. Thresholds: `<0.1` stable · `<0.25` warning · `≥0.25` drifted.
- **Rendered:** horizontal bar chart on `/ops` with threshold lines.
- **Speaker line:** "If tomorrow everyone starts using EVs, this chart lights up red. That's our signal to retrain."

### Slide F · One-Click Retrain

- **Endpoint:** `POST /train` gated by `X-Ops-Token` header; invokes `TrainPipeline.run_pipeline()` in a background thread and streams stdout back as Server-Sent Events.
- **Promotion logic:** `model_evaluation.py` promotes only if F1 beats the current registry by `MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE` (0.02). The `/ops` panel shows the final old-vs-new metric diff and whether the new model was promoted.
- **Speaker line:** Click the button live. "One click triggers the full pipeline: ingest from Mongo, validate, transform, train, track in ClearML, evaluate against the current deployed model, push to S3 if it wins. No SSH. No YAML edit. No redeploy."

---

## 6. Tech stack (slide)

| Layer | Tool | Why |
|---|---|---|
| Data store | MongoDB Atlas | Raw training data + prediction logs |
| Model registry | AWS S3 | Versioned `.pkl` + registry manifest |
| Experiment tracking | ClearML (hosted) | Hyperparams, scalars, artifacts |
| Explainability | SHAP | Per-prediction feature attribution |
| Backend | FastAPI + Uvicorn | Predict, train, drift, model info endpoints |
| Frontend | React 19 + Vite + Tailwind | Main predictor + `/ops` dashboard |
| Training | Keras (TF), scikit-learn, imblearn (SMOTE) | ANN + sklearn baselines |
| Container | Docker | Reproducible env |
| Compute | AWS EC2 + ECR | Self-hosted runner, container registry |
| CI/CD | GitHub Actions | Build → push → pull → restart |
| Drift stat | PSI (Population Stability Index) | Distribution shift detection |

---

## 7. Evaluation results (slide — fill in from `baselines.json`)

Reproduce with `python demo.py` or by clicking **Retrain** on `/ops`. Fill this table from the latest training run before presenting:

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | _ | _ | _ | _ |
| Random Forest | _ | _ | _ | _ |
| XGBoost | _ | _ | _ | _ |
| **ANN (deployed)** | _ | _ | _ | _ |

Include: confusion matrix from ClearML, ROC curve, SHAP summary plot for the deployed model, per-category slice metrics (Car / SUV / Truck / Van / Bus / Motorcycle).

---

## 8. Availability / Reliability / Efficiency / Scalability (course rubric)

- **Availability** — EC2 hosts the container behind a public URL; CI/CD redeploys in under a minute on every merge to `main`. Health endpoint returns 200 as long as the model loads.
- **Reliability** — Validation step fails the pipeline on schema violations before any training starts. Model evaluation gates promotion behind a 2% F1 improvement rule. Prediction logs persist to Mongo so we can always replay what happened.
- **Efficiency** — Preprocessor and SHAP explainer are cached on the `VehicleDataClassifier` instance (loaded once per container boot). Predictions return in < 500 ms including SHAP.
- **Scalability** — Stateless API; we can scale horizontally behind a load balancer with no code change. Training runs on the EC2 self-hosted runner independently of the serving container.

---

## 9. Limitations (be honest in the report)

- Training data is from Kaggle — synthetic distribution, not real fleet telemetry. Real-world F1 will differ.
- Drift detection uses PSI on marginal distributions; it won't catch **joint** distribution shifts (e.g., correlations between features).
- Retraining runs on a single EC2 instance — a production system would decouple training from serving.
- Confidence calibration not evaluated; `score >= 0.5` is a naive threshold. A production version would sweep the ROC curve against business cost.
- SHAP values use a sampled background set for tractability — attributions are approximate.

---

## 10. Future scope

- Scheduled retraining (cron + drift-triggered).
- Shadow-model A/B testing before promotion.
- Per-slice model monitoring (e.g., drift separately for Trucks vs. Motorcycles).
- Explainability copilot: LLM that reads the SHAP output and writes a human-readable diagnosis (*"Your vehicle is flagged because brake condition and odometer contributed most — inspect the brake pads first"*).
- Mobile-friendly PWA for in-the-field mechanics.

---

## 11. 10-minute presentation flow

| Time | Slide | Speaker | What they do |
|---|---|---|---|
| 0:00–0:45 | Title + pitch | Primary | Read section 1 + problem statement |
| 0:45–2:00 | Architecture diagram (section 3) | Primary | Walk left-to-right through the pipeline |
| 2:00–2:45 | Novelty (section 4) | Primary | Hit the five bullets fast |
| 2:45–4:30 | **Live demo: main predictor** | Primary | Scan a VIN → run prediction → point to SHAP bars |
| 4:30–6:00 | **Live demo: `/ops` page** | Primary | Walk registry → baselines table → drift chart |
| 6:00–7:30 | **Live demo: retrain** | Primary | Click Retrain → show streaming logs → show old-vs-new → switch to ClearML dashboard |
| 7:30–8:15 | Evaluation results (section 7) | Primary | Baseline table + confusion matrix |
| 8:15–8:45 | Rubric alignment (section 8) | Primary | One line per A/R/E/S |
| 8:45–9:15 | Limitations + future (sections 9–10) | Primary | Be honest, be ambitious |
| 9:15–10:00 | Contributions + Q&A | Team | Read section 12 |

**Backup:** have a pre-recorded demo video ready in case EC2 is down.

---

## 12. Contributions

- **Sankeerth B** — Data & Validation lead: Mongo ingestion, schema validation (`config/schema.yaml`), data transformation, drift snapshot emission.
- **Claude B** — Modeling & Experimentation lead: ANN architecture and training, baseline models (LR/RF/XGBoost), SHAP integration, ClearML wiring, evaluation + promotion logic.
- **Prathamesh U** — Deployment & MLOps lead: Docker, GitHub Actions CI/CD, EC2 + ECR, model registry manifest on S3, retrain endpoint with SSE, `/ops` frontend dashboard.

---

## 13. Next steps (before submission)

**Must do:**
1. **Commit and push** everything on `vin-react` — we still have uncommitted work in `app.py`, `model_pusher.py`, `model_trainer.py`, `data_transformation.py`, `src/drift.py`, `src/insights.py`, and the entire `frontend/` directory.
2. **Merge `vin-react` → `main`** and verify CI/CD redeploys to EC2.
3. **Run an end-to-end smoke test** on production:
   - main page prediction with SHAP bars ✓
   - `/ops` page renders registry, baselines, drift ✓
   - retrain button streams logs, promotes model, updates registry ✓
   - ClearML dashboard receives the new task ✓
4. **Populate `baselines.json`** by triggering one retrain so the evaluation section of the report has real numbers.
5. **Take screenshots** of: `/ops` page, SHAP bars, drift chart, ClearML dashboard, GitHub Actions run. Drop them into `plots/` for the slides.
6. **Write the 20-page report** from sections of this doc:
   - Abstract (§1), Problem (§2), Architecture (§3), Novelty (§4), Tech stack (§6), Evaluation (§7), Rubric alignment (§8), Limitations (§9), Future (§10), Contributions (§12).
   - Add: Literature review (5–10 predictive-maintenance papers), References.
7. **Record a backup demo video** (Loom or QuickTime screen recording) that matches the §11 timeline — in case EC2 is unreachable during the live demo.
8. **Build the slide deck** in PowerPoint using one slide per section of this doc — headings align one-to-one.
9. **Dry-run the demo** end-to-end twice, timing each section, so the presenter hits 10 minutes without rushing.

**Nice to have (if time permits):**
- Add a few diverse predictions (in-distribution + out-of-distribution) to seed the drift chart with visible movement before demo.
- Wire a shadow-model endpoint (`/predict?shadow=true`) that runs the candidate alongside the production model and logs disagreement rate.

**Out of scope (document as future work, don't implement):**
- LLM chatbot unless reframed as SHAP-grounded explainability copilot.
- Scheduled / drift-triggered auto-retrain.

---

## 14. Running it locally

```bash
# Dev servers
bash .dev/dev.sh           # both FE + BE
bash .dev/fe.sh            # frontend only (Vite :3000)
bash .dev/be.sh            # backend only (FastAPI :5000)

# Train from CLI
python demo.py             # triggers TrainPipeline.run_pipeline()

# Environment
cp .env.example .env       # fill in Mongo, AWS, ClearML, OPS_TOKEN
```

Required env vars (see `.env.example`):
`CONNECTION_URL`, `DB_USERNAME`, `COLLECTION_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `OPS_TOKEN`.
