from fastapi import FastAPI
from models.Pydantic_model_for_Nipah_virus import Pydantic_model_for_Nipah_virus

app = FastAPI(title="Nipah Virus Fast Apis Prediction Pipeline.")

@app.get("/")
def default():
    return {"message":"This is the Nipah Virus ML Pipeline", "Developer":"Ravi Der"}

@app.post("/nipah_prediction")
def predictoin_pipeline(user:Pydantic_model_for_Nipah_virus):
    
    