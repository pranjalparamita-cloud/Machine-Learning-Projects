from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
app = FastAPI(
    title="Heart Disease Prediction System",
    description="ML based Heart Disease Prediction",
    version="1.0"
)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
templates = Jinja2Templates(
    directory="templates"
)
model = joblib.load(
    "heart_disease_prediction_model.pkl"
)
@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )
@app.post(
    "/predict",
    response_class=HTMLResponse
)
async def predict(
    request: Request,
    age: float = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: float = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal

    }])
    prediction = model.predict(
        input_data
    )[0]
    probability = model.predict_proba(
        input_data
    )[0]


    confidence = probability[
        int(prediction)
    ] * 100
    if prediction == 1:

        result = "Heart Disease Detected"
        status = "danger"

    else:

        result = "No Heart Disease Detected"
        status = "success"

    return templates.TemplateResponse(

        "index.html",

        {
            "request": request,

            "result": result,

            "confidence": round(
                confidence,
                2
            ),

            "status": status
        }

    )