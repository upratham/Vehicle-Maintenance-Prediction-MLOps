import asyncio
import io
import json
import logging as _py_logging
import os
import queue
import threading
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import HTMLResponse, RedirectResponse, StreamingResponse
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import (
    VehicleData,
    VehicleDataClassifier,
    HyundaiCarsData,
    EngineData,
    MultiModelOrchestrator,
)
from src.pipline.training_pipeline import TrainPipeline
from src.insights import pick_service, feature_impacts
from src.drift import log_prediction, compute_drift
from src.logger import logging

MODEL_REGISTRY_PATH = os.path.join("artifact", "model_registry.json")
BASELINES_PATH = os.path.join("artifact", "baselines.json")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    DataForm class to handle and process incoming form data
    for the Vehicle Maintenance Prediction model.
    """
    def __init__(self, request: Request):
        self.request: Request = request

        # Numerical fields
        self.Reported_Issues: Optional[int] = None
        self.Vehicle_Age: Optional[int] = None
        self.Engine_Size: Optional[float] = None
        self.Odometer_Reading: Optional[float] = None
    
        self.Accident_History: Optional[int] = None
        self.Fuel_Efficiency: Optional[float] = None

        # Ordinal categorical fields
    
        self.Tire_Condition: Optional[str] = None
        self.Brake_Condition: Optional[str] = None
        self.Battery_Status: Optional[str] = None

        # Nominal categorical fields
        # Vehicle_Model : Bus, Van, SUV, Truck, Motorcycle, Car
        # Fuel_Type     : Diesel, Petrol, Electric
        # Transmission  : Manual, Automatic
        # Owner_Type    : First, Second, Third
        self.Vehicle_Model: Optional[str] = None
        self.Fuel_Type: Optional[str] = None
        self.Transmission_Type: Optional[str] = None
     

        # Date fields
   

    async def get_vehicle_data(self):
        """Retrieve and assign form data to class attributes."""
        form = await self.request.form()

        # Numerical
      
        self.Reported_Issues = form.get("Reported_Issues")
        self.Vehicle_Age = form.get("Vehicle_Age")
        self.Engine_Size = form.get("Engine_Size")
        self.Odometer_Reading = form.get("Odometer_Reading")
        self.Accident_History = form.get("Accident_History")
        self.Fuel_Efficiency = form.get("Fuel_Efficiency")

        # Ordinal categorical
        self.Tire_Condition = form.get("Tire_Condition")
        self.Brake_Condition = form.get("Brake_Condition")
        self.Battery_Status = form.get("Battery_Status")

        # Nominal categorical
        self.Vehicle_Model = form.get("Vehicle_Model")
        self.Fuel_Type = form.get("Fuel_Type")
        self.Transmission_Type = form.get("Transmission_Type")

    def get_missing_fields(self):
        """Return a list of required fields that are missing or empty."""
        required_fields = {
            "Reported_Issues": self.Reported_Issues,
            "Vehicle_Age": self.Vehicle_Age,
            "Engine_Size": self.Engine_Size,
            "Odometer_Reading": self.Odometer_Reading,
            "Accident_History": self.Accident_History,
            "Fuel_Efficiency": self.Fuel_Efficiency,
            "Tire_Condition": self.Tire_Condition,
            "Brake_Condition": self.Brake_Condition,
            "Battery_Status": self.Battery_Status,
            "Vehicle_Model": self.Vehicle_Model,
            "Fuel_Type": self.Fuel_Type,
            "Transmission_Type": self.Transmission_Type,
        }
        return [key for key, value in required_fields.items() if value in (None, "")]
  
       

@app.get("/", tags=["authentication"])
async def index(request: Request):
    """Renders the main HTML form page for vehicle data input."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "context": "Rendering"},
    )


# @app.get("/train")
# async def trainRouteClient():
#     """Endpoint to initiate the model training pipeline."""
#     try:
#         train_pipeline = TrainPipeline()
#         train_pipeline.run_pipeline()
#         return Response("Training successful!!!")
#     except Exception as e:
#         return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    """Endpoint to receive form data, process it, and make a prediction."""
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        missing_fields = form.get_missing_fields()
        if missing_fields:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "context": f"Error: Missing required fields: {', '.join(missing_fields)}",
                },
                status_code=400,
            )

        vehicle_data = VehicleData(
            
            Reported_Issues=form.Reported_Issues,
            Vehicle_Age=form.Vehicle_Age,
            Engine_Size=form.Engine_Size,
            Odometer_Reading=form.Odometer_Reading,
            Accident_History=form.Accident_History,
            Fuel_Efficiency=form.Fuel_Efficiency,
            Tire_Condition=form.Tire_Condition,
            Brake_Condition=form.Brake_Condition,
            Battery_Status=form.Battery_Status,
            Vehicle_Model=form.Vehicle_Model,
            Fuel_Type=form.Fuel_Type,
            Transmission_Type=form.Transmission_Type,
        )

        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        model_predictor = VehicleDataClassifier()
        raw_prediction = model_predictor.predict(dataframe=vehicle_df)[0]
        prediction_score = float(raw_prediction[0]) if hasattr(raw_prediction, "__len__") else float(raw_prediction)
        value = 1 if prediction_score >= 0.5 else 0

        status = "Maintenance Required" if value == 1 else "No Maintenance Needed"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "context": status},
        )

    except Exception as e:
        logging.error(f"Prediction request failed: {e}", exc_info=True)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "context": f"Error: {e}"},
            status_code=500,
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
    try:
        vehicle_data = VehicleData(**payload.model_dump())
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()
        model_predictor = VehicleDataClassifier()
        raw_prediction = model_predictor.predict(dataframe=vehicle_df)[0]
        score = float(raw_prediction[0]) if hasattr(raw_prediction, "__len__") else float(raw_prediction)
        value = 1 if score >= 0.5 else 0
        status = "Maintenance Required" if value == 1 else "No Maintenance Needed"

        raw = payload.model_dump()
        svc = pick_service(raw, score)
        impacts = feature_impacts(raw)

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
            log_prediction(raw, response)
        except Exception as log_err:
            logging.warning(f"prediction logging skipped: {log_err}")

        return JSONResponse(response)
    except Exception as e:
        logging.error(f"/predict failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ──────────────────────────────────────────────────────────────────────────────
# Hyundai cars anomaly-detection endpoint
# ──────────────────────────────────────────────────────────────────────────────

class HyundaiCarsPredictPayload(BaseModel):
    engine_temperature: float   # Engine Temperature (°C)
    brake_pad_thickness: float  # Brake Pad Thickness (mm)
    tire_pressure: float        # Tire Pressure (PSI)
    maintenance_type: str       # e.g. "Preventive", "Corrective", "Emergency"


@app.post("/predict/hyundai")
async def predict_hyundai(payload: HyundaiCarsPredictPayload):
    """Anomaly-detection prediction for Hyundai cars dataset."""
    try:
        data = HyundaiCarsData(**payload.model_dump())
        result = MultiModelOrchestrator().predict("cars_hyundai", data.get_dataframe())
        return JSONResponse(result)
    except Exception as e:
        logging.error(f"/predict/hyundai failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ──────────────────────────────────────────────────────────────────────────────
# Engine condition endpoint
# ──────────────────────────────────────────────────────────────────────────────

class EngineDataPredictPayload(BaseModel):
    engine_rpm: float        # Engine rpm
    lub_oil_pressure: float  # Lub oil pressure
    fuel_pressure: float     # Fuel pressure
    coolant_pressure: float  # Coolant pressure
    lub_oil_temp: float      # lub oil temp
    coolant_temp: float      # Coolant temp


@app.post("/predict/engine")
async def predict_engine(payload: EngineDataPredictPayload):
    """Engine condition prediction for engine_data dataset."""
    try:
        data = EngineData(**payload.model_dump())
        result = MultiModelOrchestrator().predict("engine_data", data.get_dataframe())
        return JSONResponse(result)
    except Exception as e:
        logging.error(f"/predict/engine failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ──────────────────────────────────────────────────────────────────────────────
# MLOps surface: model registry, retrain (SSE), drift
# ──────────────────────────────────────────────────────────────────────────────


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"read {path} failed: {exc}")
        return None


@app.get("/model_info")
async def model_info():
    """Expose the current model registry manifest + baselines for the Ops page."""
    manifest = _read_json(MODEL_REGISTRY_PATH) or {}
    baselines = _read_json(BASELINES_PATH) or {}
    if "baselines" not in manifest or not manifest.get("baselines"):
        manifest["baselines"] = baselines.get("models", [])
    manifest["baselines_computed_at"] = baselines.get("computed_at")
    return JSONResponse(manifest)


@app.get("/drift")
async def drift_endpoint(window: str = "7d"):
    try:
        return JSONResponse(compute_drift(window=window))
    except Exception as e:
        logging.error(f"/drift failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


class _SSEHandler(_py_logging.Handler):
    """Forward log records into a thread-safe queue for SSE streaming."""

    def __init__(self, q: "queue.Queue[str]"):
        super().__init__(level=_py_logging.INFO)
        self.q = q
        self.setFormatter(_py_logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                                datefmt="%H:%M:%S"))

    def emit(self, record: _py_logging.LogRecord) -> None:  # noqa: D401
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
        before = _read_json(MODEL_REGISTRY_PATH) or {}
        q.put_nowait("── Starting training pipeline ──")
        TrainPipeline().run_pipeline()
        after = _read_json(MODEL_REGISTRY_PATH) or {}
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


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=5000)
