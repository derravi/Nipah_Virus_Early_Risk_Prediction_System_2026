from fastapi import FastAPI

app = FastAPI(title="Nipah Virus Fast Apis Prediction Pipeline.")

@app.get("/")
def default():
    return {"message":"This is the Nipah Virus ML Pipeline", "Developer":"Ravi Der"}