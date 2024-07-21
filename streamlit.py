import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from matplotlib import style
style.use("seaborn-v0_8-darkgrid")  # Use a valid style name

import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

st.write("## Calories Burned Prediction")
st.image("https://www.healthifyme.com/blog/wp-content/uploads/2022/05/Shutterstock_1505303738-1.jpg" , use_column_width=True)
st.write(" Welcome to our WebApp, where you can easily track your predicted calorie burn. Just input your parameters—such as age, gender, BMI, and more—and discover the estimated number of kilocalories burned by your body. Enjoy a seamless and insightful experience as you monitor your fitness progress!")


st.sidebar.header("User Input Parameters : ")

def user_input_features():
    global Age , BMI , Duration , Heart_Rate , Body_Temp
    Age = st.sidebar.slider("Age : " , 10 , 100 , 20)
    BMI = st.sidebar.slider("BMI : " , 15 , 40 , 24)
    Duration = st.sidebar.slider("Duration (min) : " , 0 , 35 , 15)
    Heart_Rate = st.sidebar.slider("Heart Rate : " , 60 , 130 , 80)
    Body_Temp = st.sidebar.slider("Body Temperature (C) : " , 36 , 42 , 38)
    Gender_button = st.sidebar.radio("Gender : ", ("Male" , "Female"))

    if Gender_button == "Male":
        Gender = 1
    else:
        Gender = 0

    data = {
    "Age" : Age,
    "BMI" : BMI,
    "Duration" : Duration,
    "Heart_Rate" : Heart_Rate,
    "Body_Temp" : Body_Temp,
    "Gender" : ["Male" if Gender_button == "Male" else "Female"]
    }

    data_model = {
    "Age" : Age,
    "BMI" : BMI,
    "Duration" : Duration,
    "Heart_Rate" : Heart_Rate,
    "Body_Temp" : Body_Temp,
    "Gender_male" : Gender
    }

    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data, index=[0])
    return features , data

df , data = user_input_features()

st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

calories = pd.read_csv("C:/Users/Muskaan Jain/OneDrive/Desktop/New folder/c_data.csv")
exercise = pd.read_csv("C:/Users/Muskaan Jain/OneDrive/Desktop/New folder/e_data.csv")

exercise_df = exercise.merge(calories , on = "User_ID")
# st.write(exercise_df.head())
exercise_df.drop(columns = "User_ID" , inplace = True)

exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.3 , random_state = 1)

for data in [exercise_train_data , exercise_test_data]:         # adding BMI column to both training and test sets
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"] , 2)

exercise_train_data = exercise_train_data[["Gender" , "Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_test_data = exercise_test_data[["Gender" , "Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

st.write("R2 Score : " , round(metrics.r2_score(y_test,random_reg_prediction ) , 2))
st.write("RandomForest Root Mean Squared Error(RMSE) : " , round(np.sqrt(metrics.mean_squared_error(y_test , random_reg_prediction)) , 2))
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

st.write(round(prediction[0] , 2) , "   **kilocalories**")

st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

range = [prediction[0] - 10 , prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= range[0]) & (exercise_df["Calories"] <= range[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information : ")

boolean_age = (exercise_df["Age"] < Age).tolist()
boolean_duration = (exercise_df["Duration"] < Duration).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < Body_Temp).tolist()
boolean_heart_rate= (exercise_df["Heart_Rate"] < Heart_Rate).tolist()

st.write("You are older than %" , round(sum(boolean_age) / len(boolean_age) , 2) * 100 , "of other people.")
st.write("Your had higher exercise duration than %" , round(sum(boolean_duration) / len(boolean_duration) , 2) * 100 , "of other people.")
st.write("You had more heart rate than %" , round(sum(boolean_heart_rate) / len(boolean_heart_rate) , 2) * 100 , "of other people during exercise.")
st.write("You had higher body temperature  than %" , round(sum(boolean_body_temp) / len(boolean_body_temp) , 2) * 100 , "of other people during exercise.")
