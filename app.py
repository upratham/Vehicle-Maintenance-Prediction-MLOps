import asyncio
import json
import logging as _py_logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from uvicorn import run as app_run

from src.constants import APP_HOST
from src.drift import compute_drift, log_prediction
from src.insights import feature_impacts, pick_service
from src.logger import logging
from src.pipeline.prediction_pipeline import (
    EngineData,
    HyundaiCarsData,
    MultiModelOrchestrator,
    ProfileClassifier,
    VehicleData,
)
from src.pipeline.training_pipeline import TrainPipeline


MODEL_REGISTRY_PATH = os.path.join("artifact", "model_registry.json")
BASELINES_PATH = os.path.join("artifact", "baselines.json")

# React SPA bundle (built into frontend/dist by the Dockerfile multi-stage build,
# or by `npm run build` locally). Falls back to None when not yet built.
SPA_DIST_DIR = os.path.join("frontend", "dist")
SPA_INDEX = os.path.join(SPA_DIST_DIR, "index.html")
SPA_ASSETS = os.path.join(SPA_DIST_DIR, "assets")

app = FastAPI()

if os.path.isdir(SPA_ASSETS):
    app.mount("/assets", StaticFiles(directory=SPA_ASSETS), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictPayload(BaseModel):
    Reported_Issues: int
    Vehicle_Age: int
    Engine_Size: float
    Odometer_Reading: float
    Accident_History: int
    Fuel_Efficiency: float
    Tire_Condition: str
    Brake_Condition: str
    Battery_Status: str
    Vehicle_Model: str
    Fuel_Type: str
    Transmission_Type: str


@app.post("/predict")
async def predict_json(payload: PredictPayload):
    """JSON prediction endpoint consumed by the React frontend."""
    vehicle_data = VehicleData(**payload.model_dump())
    classifier = ProfileClassifier("vehicle_maintenance")
    raw = classifier.predict(dataframe=vehicle_data.get_vehicle_input_data_frame())[0]
    score = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
    value = 1 if score >= 0.5 else 0
    status = classifier.label(value)

    raw_payload = payload.model_dump()
    svc = pick_service(raw_payload, score)
    impacts = feature_impacts(raw_payload)

    response = {
        "status": status,
        "score": score,
        "label": value,
        "service": {
            "name": svc.service,
            "cost_low": svc.cost_low,
            "cost_high": svc.cost_high,
            "hours_low": svc.hours_low,
            "hours_high": svc.hours_high,
            "notes": svc.notes,
        },
        "impacts": [
            {"feature": f.feature, "value": str(f.value), "contribution": f.contribution}
            for f in impacts
        ],
        "explanations": [
            {
                "feature": f.feature,
                "value": str(f.value),
                "shap": round(f.contribution, 4),
                "direction": "+" if f.contribution >= 0 else "-",
            }
            for f in impacts[:5]
        ],
    }

    try:
        log_prediction(raw_payload, response)
    except Exception as log_err:
        logging.warning(f"prediction logging skipped: {log_err}")

    return JSONResponse(response)


class HyundaiCarsPredictPayload(BaseModel):
    engine_temperature: float
    brake_pad_thickness: float
    tire_pressure: float
    maintenance_type: str


@app.post("/predict/hyundai")
async def predict_hyundai(payload: HyundaiCarsPredictPayload):
    """Anomaly-detection prediction for Hyundai cars dataset."""
    data = HyundaiCarsData(**payload.model_dump())
    return JSONResponse(MultiModelOrchestrator().predict("cars_hyundai", data.get_dataframe()))


class EngineDataPredictPayload(BaseModel):
    engine_rpm: float
    lub_oil_pressure: float
    fuel_pressure: float
    coolant_pressure: float
    lub_oil_temp: float
    coolant_temp: float


@app.post("/predict/engine")
async def predict_engine(payload: EngineDataPredictPayload):
    """Engine condition prediction for engine_data dataset."""
    data = EngineData(**payload.model_dump())
    return JSONResponse(MultiModelOrchestrator().predict("engine_data", data.get_dataframe()))


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"read {path} failed: {exc}")
        return None


PROFILE_NAMES = ("vehicle_maintenance", "cars_hyundai", "engine_data")


def _resolve_profile_artifact(profile_name: str, filename: str) -> str:
    """Find filename in the latest timestamped pipeline run dir for the given profile."""
    pipeline_base = os.path.join("artifact", profile_name)
    if os.path.isdir(pipeline_base):
        runs = sorted(
            [d for d in os.listdir(pipeline_base) if os.path.isdir(os.path.join(pipeline_base, d))],
            reverse=True,
        )
        for run in runs:
            candidate = os.path.join(pipeline_base, run, filename)
            if os.path.exists(candidate):
                return candidate
    return os.path.join("artifact", filename)


def _load_json_from_s3(profile_name: str, filename: str) -> Optional[dict]:
    """Download and parse a JSON artifact file for profile_name from S3.

    Priority:
      1. Stable key co-located with the production model (written on every accepted push,
         survives artifact pruning, always reflects the live promoted model).
      2. Newest timestamped artifact run that actually contains the file (walks newest→oldest
         so a run where the model was rejected doesn't block older accepted data).
    """
    try:
        from src.cloud_storage.aws_storage import SimpleStorageService
        from src.constants import MODEL_BUCKET_NAME, ARTIFACT_S3_PREFIX, MODEL_PUSHER_S3_KEY
        s3 = SimpleStorageService()

        # 1. Stable key — model-registry/<profile>/<filename>
        try:
            stable_key = f"{MODEL_PUSHER_S3_KEY}/{profile_name}/{filename}"
            body = s3.s3_resource.Object(MODEL_BUCKET_NAME, stable_key).get()["Body"].read().decode()
            return json.loads(body)
        except Exception:
            pass  # not there yet (no accepted push for this profile), fall through

        # 2. Timestamped artifact runs, newest first
        runs = s3.list_run_prefixes(MODEL_BUCKET_NAME, f"{ARTIFACT_S3_PREFIX}/{profile_name}")
        for run_prefix in reversed(runs):
            try:
                key = f"{run_prefix}{filename}"
                body = s3.s3_resource.Object(MODEL_BUCKET_NAME, key).get()["Body"].read().decode()
                return json.loads(body)
            except Exception:
                continue  # file absent in this run (model not accepted), try older run

        return None
    except Exception as exc:
        logging.warning(f"S3 artifact fetch failed [{profile_name}/{filename}]: {exc}")
        return None


def _load_model_info(profile_name: str) -> dict:
    """Compose model registry manifest + baselines for one profile.

    Priority:
      1. MongoDB   — always current, works locally and in prod
      2. S3        — production fallback when no local artifact dir exists
      3. Local fs  — convenience for local dev
    """
    manifest = baselines = None

    # 1. MongoDB
    try:
        from src.artifact_store import load_baselines, load_model_registry
        manifest = load_model_registry(profile_name=profile_name)
        baselines = load_baselines(profile_name=profile_name)
    except Exception as e:
        logging.warning(f"artifact_store unavailable: {e}")

    # 2. S3 (production path — artifact dir won't exist in a container)
    if manifest is None:
        manifest = _load_json_from_s3(profile_name, "model_registry.json")
    if baselines is None:
        baselines = _load_json_from_s3(profile_name, "baselines.json")

    # 3. Local filesystem (local dev convenience)
    if manifest is None:
        manifest = _read_json(_resolve_profile_artifact(profile_name, "model_registry.json"))
    if baselines is None:
        baselines = _read_json(_resolve_profile_artifact(profile_name, "baselines.json"))

    manifest = manifest or {}
    baselines = baselines or {}
    manifest["profile_name"] = profile_name

    if not manifest.get("baselines"):
        manifest["baselines"] = baselines.get("models", [])
    manifest["baselines_computed_at"] = baselines.get("computed_at")

    # The winner baseline entry has the full {f1, precision, recall, roc_auc, accuracy} set;
    # the pipeline-written metrics dict only has f1 scalars, so backfill from the winner.
    m = manifest.get("metrics", {})
    if "f1" not in m:
        winner = next((b for b in manifest.get("baselines", []) if b.get("winner")), None)
        if winner:
            manifest["metrics"] = {k: winner.get(k) for k in ["f1", "precision", "recall", "roc_auc", "accuracy"]}

    return manifest


@app.get("/model_info")
async def model_info(profile: Optional[str] = None):
    """Return one profile's model registry manifest + baselines (default: vehicle_maintenance)."""
    return JSONResponse(_load_model_info(profile or "vehicle_maintenance"))


@app.get("/model_info/all")
async def model_info_all():
    """Return registry+baselines for every known profile, keyed by profile name."""
    return JSONResponse({name: _load_model_info(name) for name in PROFILE_NAMES})


@app.get("/drift")
async def drift_endpoint(window: str = "7d"):
    return JSONResponse(compute_drift(window=window))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Translate any unhandled exception into a JSON 500. HTTPException uses its own handler."""
    logging.error(f"{request.method} {request.url.path} failed: {exc}", exc_info=True)
    return JSONResponse({"error": str(exc)}, status_code=500)


class _SSEHandler(_py_logging.Handler):
    """Forward log records into a thread-safe queue for SSE streaming."""

    def __init__(self, q: "queue.Queue[str]"):
        super().__init__(level=_py_logging.INFO)
        self.q = q
        self.setFormatter(_py_logging.Formatter(
            "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S",
        ))

    def emit(self, record: _py_logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


def _run_training(q: "queue.Queue[str]") -> None:
    """Run the training pipeline on a worker thread, piping logs into the queue."""
    root = _py_logging.getLogger()
    handler = _SSEHandler(q)
    root.addHandler(handler)
    try:
        from src.artifact_store import load_model_registry
        before = load_model_registry() or _read_json(MODEL_REGISTRY_PATH) or {}
        q.put_nowait("── Starting training pipeline ──")
        TrainPipeline().run_pipeline()
        after = load_model_registry() or _read_json(MODEL_REGISTRY_PATH) or {}
        summary = {
            "event": "done",
            "promoted": bool(after.get("promoted")),
            "old_metrics": before.get("metrics", {}),
            "new_metrics": after.get("metrics", {}),
            "new_version": after.get("version"),
        }
        q.put_nowait("__SUMMARY__:" + json.dumps(summary))
    except Exception as exc:
        q.put_nowait(f"ERROR: {exc}")
        q.put_nowait("__SUMMARY__:" + json.dumps({"event": "error", "error": str(exc)}))
    finally:
        q.put_nowait("__DONE__")
        root.removeHandler(handler)


@app.post("/train")
async def train_endpoint(x_ops_token: Optional[str] = Header(default=None)):
    """Gate on OPS_TOKEN header; stream training logs via SSE; return metric diff."""
    expected = os.getenv("OPS_TOKEN")
    if expected and x_ops_token != expected:
        raise HTTPException(status_code=401, detail="invalid ops token")

    q: "queue.Queue[str]" = queue.Queue()
    worker = threading.Thread(target=_run_training, args=(q,), daemon=True)
    worker.start()

    def _next(timeout: float) -> Optional[str]:
        try:
            return q.get(timeout=timeout)
        except Exception:
            return None

    async def event_stream():
        yield f": retrain started at {datetime.now(timezone.utc).isoformat()}\n\n"
        while True:
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, _next, 1.0)
            if line is None:
                if not worker.is_alive() and q.empty():
                    break
                yield ": keepalive\n\n"
                continue
            if line == "__DONE__":
                yield "event: done\ndata: {}\n\n"
                break
            if line.startswith("__SUMMARY__:"):
                yield f"event: summary\ndata: {line[len('__SUMMARY__:'):]}\n\n"
                continue
            yield f"data: {json.dumps({'line': line})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """SPA catch-all: serve index.html for any unmatched GET so client-side routing works.
    All API routes are registered above this and will match first."""
    if os.path.exists(SPA_INDEX):
        return FileResponse(SPA_INDEX)
    return JSONResponse(
        {"error": "Frontend bundle not built. Run `cd frontend && npm run build` or build via Docker."},
        status_code=503,
    )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=8000)
