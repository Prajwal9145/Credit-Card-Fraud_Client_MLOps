from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import (CreditCard,
                                             CreditCardDataClassifier)
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.LIMIT_BAL: Optional[int] = None
        self.SEX: Optional[int] = None
        self.EDUCATION: Optional[int] = None
        self.MARRIAGE: Optional[int] = None
        self.AGE: Optional[int] = None
        self.PAY_0: Optional[int] = None
        self.PAY_2: Optional[int] = None
        self.PAY_3: Optional[int] = None
        self.PAY_4: Optional[int] = None
        self.PAY_5: Optional[int] = None
        self.PAY_6: Optional[int] = None
        self.BILL_AMT1: Optional[int] = None
        self.BILL_AMT2: Optional[int] = None
        self.BILL_AMT3: Optional[int] = None
        self.BILL_AMT4: Optional[int] = None
        self.BILL_AMT5: Optional[int] = None
        self.BILL_AMT6: Optional[int] = None
        self.PAY_AMT1: Optional[int] = None
        self.PAY_AMT2: Optional[int] = None
        self.PAY_AMT3: Optional[int] = None
        self.PAY_AMT4: Optional[int] = None
        self.PAY_AMT5: Optional[int] = None
        self.PAY_AMT6: Optional[int] = None
                

    async def get_creditcard_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.LIMIT_BAL = form.get("LIMIT_BAL")
        self.SEX = form.get("SEX")
        self.EDUCATION = form.get("EDUCATION")
        self.MARRIAGE = form.get("MARRIAGE")
        self.AGE = form.get("AGE")
        self.PAY_0 = form.get("PAY_0")
        self.PAY_2 = form.get("PAY_2")
        self.PAY_3 = form.get("PAY_3")
        self.PAY_4 = form.get("PAY_4")
        self.PAY_5 = form.get("PAY_5")
        self.PAY_6 = form.get("PAY_6")
        self.BILL_AMT1 = form.get("BILL_AMT1")
        self.BILL_AMT2 = form.get("BILL_AMT2")
        self.BILL_AMT3 = form.get("BILL_AMT3")
        self.BILL_AMT4 = form.get("BILL_AMT4")
        self.BILL_AMT5 = form.get("BILL_AMT5")
        self.BILL_AMT6 = form.get("BILL_AMT6")
        self.PAY_AMT1 = form.get("PAY_AMT1")
        self.PAY_AMT2 = form.get("PAY_AMT2")
        self.PAY_AMT3 = form.get("PAY_AMT3")
        self.PAY_AMT4 = form.get("PAY_AMT4")
        self.PAY_AMT5 = form.get("PAY_AMT5")
        self.PAY_AMT6 = form.get("PAY_AMT6")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "index.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_creditcard_data()
        
        creditcard = CreditCard(
                                LIMIT_BAL= form.LIMIT_BAL,
                                SEX= form.SEX,
                                EDUCATION= form.EDUCATION,
                                MARRIAGE= form.MARRIAGE,
                                AGE= form.AGE,
                                PAY_0= form.PAY_0,
                                PAY_2= form.PAY_2,
                                PAY_3= form.PAY_3,
                                PAY_4= form.PAY_4,
                                PAY_5= form.PAY_5,
                                PAY_6= form.PAY_6,
                                BILL_AMT1= form.BILL_AMT1,
                                BILL_AMT2= form.BILL_AMT2,
                                BILL_AMT3= form.BILL_AMT3,
                                BILL_AMT4= form.BILL_AMT4,
                                BILL_AMT5= form.BILL_AMT5,
                                BILL_AMT6= form.BILL_AMT6,
                                PAY_AMT1 = form.PAY_AMT1,
                                PAY_AMT2 = form.PAY_AMT2,
                                PAY_AMT3 = form.PAY_AMT3,
                                PAY_AMT4 = form.PAY_AMT4,
                                PAY_AMT5 = form.PAY_AMT5,
                                PAY_AMT6 = form.PAY_AMT6,
                                )

        # Convert form data into a DataFrame for the model
        vehicle_df = creditcard.get_credit_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = CreditCardDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=vehicle_df)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "Response-Yes" if value == 1 else "Response-No"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)