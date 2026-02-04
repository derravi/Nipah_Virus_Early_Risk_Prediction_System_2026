from fastapi import FastAPI
from models.Pydantic_model_for_Nipah_virus import Pydantic_model_for_Nipah_virus
import pandas as pd
import numpy as np
import pickle
from main import lowercase,encod_field

app = FastAPI(title="Nipah Virus Fast Apis Prediction Pipeline.")

with open("/models/pickle_model_of_Nipah_virus_pipelines.pkl","rb") as f:
    model = pickle.load(f)

temp2 = model['temp2']
encoder = model['encoder']
lb = model['lbelencoder']
std = model['standardscaler']
lr = model['logisticregression']
xgb = model['xgbooster']


@app.get("/")
def default():
    return {"message":"This is the Nipah Virus ML Pipeline", "Developer":"Ravi Der"}

@app.post("/nipah_prediction")
def predictoin_pipeline(user:Pydantic_model_for_Nipah_virus):
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
lowercase(temp2)

#Lets scal down the all the datasets.
encod_field(data)