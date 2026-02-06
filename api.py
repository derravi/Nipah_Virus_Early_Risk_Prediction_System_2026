from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models.Pydantic_model_for_Nipah_virus import Nipah_Userinput
import pandas as pd
import numpy as np
import pickle

try:
    with open("models/pickle_model_of_Nipah_virus_pipelines.pkl","rb") as f:
        model = pickle.load(f)

    temp2 = model['temp2']
    encoder = model['encoder']
    lb = model['lbelencoder']
    std = model['standardscaler']
    lr = model['logisticregression']
    xgb = model['xgbooster']

except FileNotFoundError as e:
    print(f"Error : {e}")

app = FastAPI(title="Nipah Virus Fast Apis Prediction Pipeline.")

@app.get("/")
def default():
    return {"message":"This is the Nipah Virus ML Pipeline", "Developer":"Ravi Der"}

@app.post("/nipah_prediction")
def predictoin_pipeline(user:Nipah_Userinput):
    data = pd.DataFrame([{
    "age":user.age,
    "gender":user.gender,
    "temperature":user.temperature,
    "oxygen_level":user.oxygen_level,
    "heart_rate":user.heart_rate,
    "days_sick":user.days_sick,
    "fever":user.fever,
    "headache":user.headache,
    "cough":user.cough,
    "throat_pain":user.throat_pain,
    "vomiting":user.vomiting,
    "breathing_problem":user.breathing_problem,
    "body_pain":user.body_pain,
    "confusion_sleepy":user.confusion_sleepy,
    "fits":user.fits,
    "contact_patient":user.contact_patient,
    "outbreak_visit":user.outbreak_visit,
    "bat_or_fruit_contact":user.bat_or_fruit_contact,
    "date_sap_drink":user.date_sap_drink,
    "pig_or_bat_contact":user.pig_or_bat_contact
    }])

    #Convert the Categorical data into the lower charecter.
    def lowercase1(low):
        for i in low:
            data[i] = data[i][0].lower()
    lowercase1(temp2)

    #Lets scal down the all the datasets.
    def encod_field1(encod):
        for i in encod:
            data[i] = encoder[i].transform(data[i])
    encod_field1(temp2)

    #Lets Use the StandardScaler for the Scal Down the data
    scaled_data = std.transform(data)

    #Lets Predict the Models Results.

    #Using the LogisticRegressor
    lr_prediction = int(round(lr.predict(scaled_data)[0],0))

    #Using XGBooster
    xgb_prediction = int(round(xgb.predict(scaled_data)[0],0))

    lr_prediction_result ="Patient is likely to have Nipah Virus." if lr_prediction == 1 else "Patient is unlikely to have Nipah Virus."

    xgb_prediction_result = "Patient is likely to have Nipah Virus." if xgb_prediction == 1 else "Patient is unlikely to have Nipah Virus."

    return JSONResponse(status_code=200, 
                        content={
                                "Logistic_Regressor Model Prediction":f"{lr_prediction_result}",
                                "XGBooster Model Prediction":f"{xgb_prediction_result}"
                        })