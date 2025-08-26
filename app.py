# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from uvicorn import run as app_run

from insurance_src.constants import APP_HOST, APP_PORT
from insurance_src.pipeline.training_pipeline import TrainPipeline
from insurance_src.pipeline.prediction_pipeline import InsuranceData, InsuranceClassifier

# Create FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request = request
        # Define all fields with default None
        self.months_as_customer = None
        self.age = None
        self.policy_state = None
        self.policy_csl = None
        self.policy_deductable = None
        self.policy_annual_premium = None
        self.umbrella_limit = None
        self.insured_sex = None
        self.insured_education_level = None
        self.insured_occupation = None
        self.insured_hobbies = None
        self.insured_relationship = None
        self.capital_gains = None
        self.capital_loss = None
        self.incident_type = None
        self.collision_type = None
        self.incident_severity = None
        self.authorities_contacted = None
        self.incident_state = None
        self.incident_hour_of_the_day = None
        self.number_of_vehicles_involved = None
        self.property_damage = None
        self.bodily_injuries = None
        self.witnesses = None
        self.police_report_available = None
        self.total_claim_amount = None
        self.injury_claim = None
        self.property_claim = None
        self.vehicle_claim = None
        self.auto_make = None
        self.auto_year = None
        
    async def get_insurance_data(self):
        form = await self.request.form()
        # Safely parse each field, with type conversion
        self.months_as_customer = int(form.get("months_as_customer"))
        self.age = int(form.get("age"))
        self.policy_state = form.get("policy_state")
        self.policy_csl = form.get("policy_csl")
        self.policy_deductable = int(form.get("policy_deductable"))
        self.policy_annual_premium = float(form.get("policy_annual_premium"))
        self.umbrella_limit = int(form.get("umbrella_limit"))
        self.insured_sex = form.get("insured_sex")
        self.insured_education_level = form.get("insured_education_level")
        self.insured_occupation = form.get("insured_occupation")
        self.insured_hobbies = form.get("insured_hobbies")
        self.insured_relationship = form.get("insured_relationship")
        self.capital_gains = int(form.get("capital_gains"))
        self.capital_loss = int(form.get("capital_loss"))
        self.incident_type = form.get("incident_type")
        self.collision_type = form.get("collision_type")
        self.incident_severity = form.get("incident_severity")
        self.authorities_contacted = form.get("authorities_contacted")
        self.incident_state = form.get("incident_state")
        self.incident_hour_of_the_day = int(form.get("incident_hour_of_the_day"))
        self.number_of_vehicles_involved = int(form.get("number_of_vehicles_involved"))
        self.property_damage = form.get("property_damage")
        self.bodily_injuries = int(form.get("bodily_injuries"))
        self.witnesses = int(form.get("witnesses"))
        self.police_report_available = form.get("police_report_available")
        self.total_claim_amount = int(form.get("total_claim_amount"))
        self.injury_claim = int(form.get("injury_claim"))
        self.property_claim = int(form.get("property_claim"))
        self.vehicle_claim = int(form.get("vehicle_claim"))
        self.auto_make = form.get("auto_make")
        self.auto_year = int(form.get("auto_year"))


@app.get("/", tags=["UI"])
async def index(request: Request):
    # Render form page
    return templates.TemplateResponse("insurance.html", {"request": request, "context": ""})


@app.get("/train")
async def train_route():
    try:
        # train_pipeline = TrainPipeline()
        # train_pipeline.run_pipeline()
        # return Response("Training successful !!")
        # os.system("python main.py")
        os.system("dvc repro")
        return "Training done successfully!"
    except Exception as e:
        import traceback
        return {"status": False, "error": traceback.format_exc()}



@app.post("/submit_insurance_data", tags=["Prediction"])
async def submit_insurance_data(request: Request):
    try:
        form = DataForm(request)
        await form.get_insurance_data()
        
        insurance_data = InsuranceData(
                    months_as_customer=form.months_as_customer or 0,
                    age=form.age or 0,
                    policy_state=form.policy_state or "",
                    policy_csl=form.policy_csl or "",
                    policy_deductable=form.policy_deductable or 0,
                    policy_annual_premium=form.policy_annual_premium or 0.0,
                    umbrella_limit=form.umbrella_limit or 0,
                    insured_sex=form.insured_sex or "",
                    insured_education_level=form.insured_education_level or "",
                    insured_occupation=form.insured_occupation or "",
                    insured_hobbies=form.insured_hobbies or "",
                    insured_relationship=form.insured_relationship or "",
                    capital_gains=form.capital_gains or 0,
                    capital_loss=form.capital_loss or 0,
                    incident_type=form.incident_type or "",
                    collision_type=form.collision_type or "",
                    incident_severity=form.incident_severity or "",
                    authorities_contacted=form.authorities_contacted or "",
                    incident_state=form.incident_state or "",
                    incident_hour_of_the_day=form.incident_hour_of_the_day or 0,
                    number_of_vehicles_involved=form.number_of_vehicles_involved or 0,
                    property_damage=form.property_damage or "",
                    bodily_injuries=form.bodily_injuries or 0,
                    witnesses=form.witnesses or 0,
                    police_report_available=form.police_report_available or "",
                    total_claim_amount=form.total_claim_amount or 0,
                    injury_claim=form.injury_claim or 0,
                    property_claim=form.property_claim or 0,
                    vehicle_claim=form.vehicle_claim or 0,
                    auto_make=form.auto_make or "",
                    auto_year=form.auto_year or 0,
        )

        insurance_df = insurance_data.get_insurance_input_data_frame()
        model_predictor = InsuranceClassifier()
        prediction = model_predictor.predict(dataframe=insurance_df)[0]
        result = "Yes" if prediction == 1 else "No"

        return templates.TemplateResponse(
            "insurance.html",
            {"request": request, "context": result},
        )

    except Exception as e:
        import traceback
        return {"status": False, "error": traceback.format_exc()}





if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
