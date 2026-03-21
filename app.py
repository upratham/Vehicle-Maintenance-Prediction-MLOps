from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline

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
        self.Mileage: Optional[float] = None
        self.Reported_Issues: Optional[int] = None
        self.Vehicle_Age: Optional[int] = None
        self.Engine_Size: Optional[float] = None
        self.Odometer_Reading: Optional[float] = None
        self.Insurance_Premium: Optional[float] = None
        self.Service_History: Optional[int] = None
        self.Accident_History: Optional[int] = None
        self.Fuel_Efficiency: Optional[float] = None

        # Ordinal categorical fields
        self.Maintenance_History: Optional[str] = None
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
        self.Owner_Type: Optional[str] = None

        # Date fields
        self.Last_Service_Date: Optional[str] = None
        self.Warranty_Expiry_Date: Optional[str] = None

    async def get_vehicle_data(self):
        """Retrieve and assign form data to class attributes."""
        form = await self.request.form()

        # Numerical
        self.Mileage = form.get("Mileage")
        self.Reported_Issues = form.get("Reported_Issues")
        self.Vehicle_Age = form.get("Vehicle_Age")
        self.Engine_Size = form.get("Engine_Size")
        self.Odometer_Reading = form.get("Odometer_Reading")
        self.Insurance_Premium = form.get("Insurance_Premium")
        self.Service_History = form.get("Service_History")
        self.Accident_History = form.get("Accident_History")
        self.Fuel_Efficiency = form.get("Fuel_Efficiency")

        # Ordinal categorical
        self.Maintenance_History = form.get("Maintenance_History")
        self.Tire_Condition = form.get("Tire_Condition")
        self.Brake_Condition = form.get("Brake_Condition")
        self.Battery_Status = form.get("Battery_Status")

        # Nominal categorical
        self.Vehicle_Model = form.get("Vehicle_Model")
        self.Fuel_Type = form.get("Fuel_Type")
        self.Transmission_Type = form.get("Transmission_Type")
        self.Owner_Type = form.get("Owner_Type")

        # Dates
        self.Last_Service_Date = form.get("Last_Service_Date")
        self.Warranty_Expiry_Date = form.get("Warranty_Expiry_Date")


@app.get("/", tags=["authentication"])
async def index(request: Request):
    """Renders the main HTML form page for vehicle data input."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "context": "Rendering"}
    )


@app.get("/train")
async def trainRouteClient():
    """Endpoint to initiate the model training pipeline."""
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    """Endpoint to receive form data, process it, and make a prediction."""
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
            Mileage=form.Mileage,
            Reported_Issues=form.Reported_Issues,
            Vehicle_Age=form.Vehicle_Age,
            Engine_Size=form.Engine_Size,
            Odometer_Reading=form.Odometer_Reading,
            Insurance_Premium=form.Insurance_Premium,
            Service_History=form.Service_History,
            Accident_History=form.Accident_History,
            Fuel_Efficiency=form.Fuel_Efficiency,
            Maintenance_History=form.Maintenance_History,
            Tire_Condition=form.Tire_Condition,
            Brake_Condition=form.Brake_Condition,
            Battery_Status=form.Battery_Status,
            Vehicle_Model=form.Vehicle_Model,
            Fuel_Type=form.Fuel_Type,
            Transmission_Type=form.Transmission_Type,
            Owner_Type=form.Owner_Type,
            Last_Service_Date=form.Last_Service_Date,
            Warranty_Expiry_Date=form.Warranty_Expiry_Date,
        )

        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        model_predictor = VehicleDataClassifier()
        value = model_predictor.predict(dataframe=vehicle_df)[0]

        status = "Maintenance Required" if value == 1 else "No Maintenance Needed"

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)