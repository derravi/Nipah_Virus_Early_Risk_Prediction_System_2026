import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle

try:
    df = pd.read_csv("Nipah_Virus_dataset_File/cleaned_data.csv")
    df.head()
except FileExistsError as e:
    print(f"e")

#check there is any null values or not.
print("Lets see the Null values of the all columns.........")
df.isna().sum()

#Remove one unwanted column.
print("Drop one Unwanted Column............")
df.drop(columns=["Unnamed: 0"],inplace=True)

#Lets check there is have any OutLier or not.
print("Lets see there is anu outliers or not...................")
#make one list of the all the columns.
column_list = ['age','temperature', 'oxygen_level', 'heart_rate',
       'days_sick','nipah_result']

plt.figure(figsize=(8,4))
for i in column_list:
    plt.boxplot(df[i], vert=False)
    plt.title(f"{i} Column")
    plt.savefig(f"OutLiers_graphs/Before_ourliers/{i}_outliers.png",dpi=100,bbox_inches='tight')
    plt.show()

#Lets remove the OutLiers from this datasets.
#We are use the IQR method for this.

column_list_for_outliers = ['age','temperature', 'oxygen_level', 'heart_rate']

for i in column_list_for_outliers:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)

    IQR = q3-q1

    min_range = q1 - (1.5*IQR)
    max_range = q3 + (1.5*IQR)

    df = df[(df[i] >= min_range) & (df[i] <= max_range)]
    

#Lets Recheck the OutLiers of the all the columns.

column_list = ['age','temperature', 'oxygen_level', 'heart_rate',
       'days_sick','nipah_result']

plt.figure(figsize=(8,4))
for i in column_list:
    plt.boxplot(df[i], vert=False)
    plt.title(f"{i} Column")
    plt.savefig(f"OutLiers_graphs/After_emoving_Outliers/{i}_removed_outliers.png",dpi=100,bbox_inches='tight')
    plt.show()

#Lets convert the All the Categorical data into the numaric formate.
#WE are scal down of this all the columns.

temp2 = []
temp3 = []

for i in df.columns:
    if df[i].dtype == "object":
        temp2.append(i)
    else:
        temp3.append(i)


#Convert the Categorical to numaric. Using the LabelEncoder We are encode the all the categoricle data
encoder = {}

for i in temp2:
    lb = LabelEncoder()
    df[i] = lb.fit_transform(df[i])
    encoder[i] = lb
print("Dat Encoding Is Completed....")

#Lets Use th train test Split for the Data

x = df.iloc[:,:-1]
y = df["nipah_result"]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.20)

print("The train test split and the column division is completed.....")

#Lets scal down all the columns and make the data into like the machine understandable formate.

std = StandardScaler()

x_train_scaled = std.fit_transform(x_train)
x_test_scaled = std.transform(x_test)

print("The Scaling of the datset is now conpleted................")

#Use the Logistic Regressor Model

lr = LogisticRegression()

lr.fit(x_train_scaled,y_train)
y_lr_predicted = lr.predict(x_test_scaled)

print("Logistic Regressor model trained for the given datasets.")

#Check the Rehressor Model Performance using some mathamatics outcomes.
print("Lets see the Ligistic Regression model performance.")
print("Mean Absolute Error(MAE):",round(mean_absolute_error(y_test,y_lr_predicted),4))
print("Mean Square Error(MSE):",round(mean_squared_error(y_test,y_lr_predicted),4))
print("Root Mean Square Error(MSE):",round(np.square(mean_squared_error(y_test,y_lr_predicted)),4))
print("R2 Score:",round(r2_score(y_test,y_lr_predicted),4))

#Lets use XGBooster Model for this

xgb = XGBRegressor(
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=9
)
xgb.fit(x_train_scaled,y_train)
y_xgb_predicted = xgb.predict(x_test_scaled)

print("XGBooster model trained for the given datasets.")


print("Lets see the XGBooster model performance.")
print("Mean Absolute Error(MAE):",round(mean_absolute_error(y_test,y_xgb_predicted),4))
print("Mean Square Error(MSE):",round(mean_squared_error(y_test,y_xgb_predicted),4))
print("Root Mean Square Error(MSE):",round(np.square(mean_squared_error(y_test,y_xgb_predicted)),4))
print("R2 Score:",round(r2_score(y_test,y_xgb_predicted),4))

#User Input 

print("Lets take the user input and find the new prediction on the new data.")
#Lets take the user input
# age = int(input("Enter Age: "))
# gender = input("Enter Gender (Male/Female/Other): ")
# temperature = float(input("Enter Body Temperature (°C): "))
# oxygen_level = float(input("Enter Oxygen Level (%): "))
# heart_rate = float(input("Enter Heart Rate (bpm): "))
# days_sick = float(input("Enter Number of Days Sick: "))
# fever = input("Do you have fever? (Yes/No): ")
# headache = input("Do you have headache? (Yes/No): ")
# cough = input("Do you have cough? (Yes/No): ")
# throat_pain = input("Do you have throat pain? (Yes/No): ")
# vomiting = input("Do you have vomiting? (Yes/No): ")
# breathing_problem = input("Do you have breathing problem? (Yes/No): ")
# body_pain = input("Do you have body pain? (Yes/No): ")
# confusion_sleepy = input("Do you feel confusion or sleepiness? (Yes/No): ")
# fits = input("Do you have fits/seizures? (Yes/No): ")
# contact_patient = input("Have you contacted a Nipah patient? (Yes/No): ")
# outbreak_visit = input("Visited outbreak area recently? (Yes/No): ")
# bat_or_fruit_contact = input("Contact with bat or bitten fruits? (Yes/No): ")
# date_sap_drink = input("Consumed date palm sap? (Yes/No): ")
# pig_or_bat_contact = input("Contact with pig or bat? (Yes/No): ")

age = 28
gender = "Male"
temperature = 102.5          
oxygen_level = 92.0         
heart_rate = 110.0           
days_sick = 4.0
fever = "Yes"
headache = "Yes"
cough = "Yes"
throat_pain = "No"
vomiting = "No"
breathing_problem = "Yes"
body_pain = "Yes"
confusion_sleepy = "No"
fits = "No"
contact_patient = "Yes"
outbreak_visit = "No"
bat_or_fruit_contact = "Yes"
date_sap_drink = "No"
pig_or_bat_contact = "No"


#Lets make the DataFrame for the given user input.
new_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "temperature": temperature,
    "oxygen_level": oxygen_level,
    "heart_rate": heart_rate,
    "days_sick": days_sick,
    "fever": fever,
    "headache": headache,
    "cough": cough,
    "throat_pain": throat_pain,
    "vomiting": vomiting,
    "breathing_problem": breathing_problem,
    "body_pain": body_pain,
    "confusion_sleepy": confusion_sleepy,
    "fits": fits,
    "contact_patient": contact_patient,
    "outbreak_visit": outbreak_visit,
    "bat_or_fruit_contact": bat_or_fruit_contact,
    "date_sap_drink": date_sap_drink,
    "pig_or_bat_contact": pig_or_bat_contact
}])

#Lets make the perfect data

#Convert the all the Values into the Lower Case because we are Train the Model into this types of the datasets.
def lowercase(low):
    for i in low:
        new_data[i] = new_data[i][0].lower()
    print("Lower Charecter is completed.")
lowercase(temp2)

#Lets Encode the userdata.
def encod_field(encod):
    for i in encod:
        new_data[i] = encoder[i].transform(new_data[i])
    print("Label data encoding completed.")
encod_field(temp2)

#Lets ScalDown the all the data.
scaled_data = std.transform(new_data)
print("We are convert the labeld data into the Numaric formate.")

#Lets make the UserInput Prediction Pipelines.

lr_userinput_prediction = int(round(lr.predict(new_data)[0],0))
xgb_userinput_prediction = int(round(xgb.predict(new_data)[0],0))

print("\nFinal Predictions")
print(f"Logistic Regression says:{lr_userinput_prediction}")
print(f"XGBoost says:{xgb_userinput_prediction}")

print("\nFinal Decision based on Logistic Regression")
if lr_userinput_prediction == 1:
    print("Patient is likely to have Nipah Virus.")
else:
    print("Patient is unlikely to have Nipah Virus.")

print("\nFinal Decision based on XGBoost")
if xgb_userinput_prediction == 1:
    print("Patient is likely to have Nipah Virus.")
else:
    print("Patient is unlikely to have Nipah Virus.")

#Lets Make the Pickle Model for this ML Pipelines.

model= {
    "temp2":temp2,
    "encoder":encoder,
    "lbelencoder":lb,
    "standardscaler":std,
    "logisticregression":lr,
    "xgbooster":xgb
}

with open("models/pickle_model_of_Nipah_virus_pipelines.pkl","wb") as f:
    pickle.dump(model,f)
