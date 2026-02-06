from pydantic import BaseModel,Field
from typing import Annotated


class Nipah_Userinput(BaseModel):
    age:Annotated[int,Field(...,ge=0,description="Enter the Age of the Patient.",examples=[24])]
    gender:Annotated[str,Field(...,description="Enter Gender (Male/Female/Other):",examples=["Male"])]
    temperature:Annotated[float,Field(...,ge=0,description="Enter Body Temperature (°C):",examples=[102.5])]
    oxygen_level:Annotated[float,Field(...,ge=0,description="Enter Oxygen Level (%):",examples=[92.0])]
    heart_rate:Annotated[float,Field(...,gt=0,description="Enter Heart Rate (bpm):",examples=[110.0])]
    days_sick:Annotated[float,Field(...,ge=0,description="Enter Number of Days Sick:",examples=[4.0])]
    fever:Annotated[str,Field(...,description="Do you have fever? (Yes/No):",examples=["Yes"])]
    headache:Annotated[str,Field(...,description="Do you have headache? (Yes/No):",examples=["Yes"])]
    cough:Annotated[str,Field(...,description="Do you have cough? (Yes/No):",examples=["Yes"])]
    throat_pain:Annotated[str,Field(...,description="Do you have throat pain? (Yes/No):",examples=["No"])]
    vomiting:Annotated[str,Field(...,description="Do you have vomiting? (Yes/No):",examples=["No"])]
    breathing_problem:Annotated[str,Field(...,description="Do you have breathing problem? (Yes/No):",examples=["Yes"])]
    body_pain:Annotated[str,Field(...,description="Do you have body pain? (Yes/No):",examples=["Yes"])]
    confusion_sleepy:Annotated[str,Field(...,description="Do you feel confusion or sleepiness? (Yes/No):",examples=["No"])]
    fits:Annotated[str,Field(...,description="Do you have fits/seizures? (Yes/No):",examples=["No"])]
    contact_patient:Annotated[str,Field(...,description="Have you contacted a Nipah patient? (Yes/No):",examples=["Yes"])]
    outbreak_visit:Annotated[str,Field(...,description="Visited outbreak area recently? (Yes/No):",examples=["No"])]
    bat_or_fruit_contact:Annotated[str,Field(...,description="Contact with bat or bitten fruits? (Yes/No):",examples=["Yes"])]
    date_sap_drink:Annotated[str,Field(...,description="Consumed date palm sap? (Yes/No):",examples=["No"])]
    pig_or_bat_contact:Annotated[str,Field(...,description="Contact with pig or bat? (Yes/No):",examples=["No"])]