import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#page navigation
st.set_page_config(page_title= "DIABETES PREDICTION",
                   layout="wide",
                   initial_sidebar_state= "expanded"
                  )

#sidebar option set up
select_page= option_menu(None,["Home", "DiabetesPrediction","Evaluation"],
                icons=["house","graph-up-arrow", "exclamation-circle","bar_chart"],
                menu_icon= "menu-button-wide",
                orientation="horizontal",
                default_index=0,
                styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "#6F36AD"},
                    "nav-link-selected": {"background-color": "#6F36AD"}})

#setting up home page
if select_page == "Home":

    col1,col2,col3= st.columns(3)
               
    with col2:
        st.title(":black[*Diabetes Prediction using streamlit*]")
        st.write(":green[Technologies used :]  Python, Pandas, sklearn, Streamlit, and DecisionTreeClassifier")
        st.write(":green[Overview:] This application provides a comprehensive understanding of the Diabetes Prediction Application, its features, functionality, and purpose. Users can utilize this application to gain insights into their diabetes risk and take proactive measures towards better health management. ")
        st.caption(":green[######*Application created by Subbulakshmi*######]")
        
# Load the dataset
df = pd.read_csv("C:\\Users\\Admin\\Desktop\\myproject\\diabetes_prediction_dataset.csv")

# Encode categorical features
encoder = OrdinalEncoder()
df[['gender']] = encoder.fit_transform(df[['gender']])

# Define custom encoding for smoking history
smoking_history_encoding = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
df['smoking_history'] = df['smoking_history'].map(smoking_history_encoding)

# Split data into features and target
x = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingClassifier()
model.fit(x_train, y_train)

def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
    prediction = model.predict(input_data)
    return prediction[0]

#prediction page
if select_page == "DiabetesPrediction":
    st.info("""
            *FILL OUT THE FOLLOWING INFORMATION TO PREDICT WHETHER YOU HAVE DIABETES OR NOT!!!
           INFORMATION PROVIDED WILL NOT BE SHARED*
            
            """,icon="🚨")
    col4,col5,col6= st.columns(3)
    
    with col4:
        name= st.text_input("Enter Your Name below")
        age = st.slider("Age", min_value=1, max_value=100, step=1)
        smoking_history = st.selectbox("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
    
        
    with col6:    
        bmi = st.number_input("BMI")
        HbA1c_level = st.number_input("HbA1c Level")
        blood_glucose_level = st.number_input("Blood Glucose Level")
        
     
    with col5:
        gender = st.radio("Gender", ["Male", "Female"])    
        hypertension = st.radio("Hypertension (1 for Yes, 0 for No)", [0, 1])
        heart_disease = st.radio("Heart Disease (1 for Yes, 0 for No)", [0, 1])   

        if st.button("Predict"):
            gender_encoded = 1 if gender == "Male" else 0
            smoking_history_encoded = smoking_history_encoding[smoking_history]
            prediction = predict_diabetes(gender_encoded, age, hypertension, heart_disease, smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level)
            if prediction == 1:
                st.warning(f"Sorry {name}! Based on the provided information, you are predicted to have diabetes",icon="🚨")
            else:
                st.success(f"Congrats {name}! Based on the provided information, you are predicted to not have diabetes")
                st.balloons()

if select_page == "Evaluation":
    st.subheader(":blue[Model Evaluation]")
    st.caption("This model uses GradientBooster classifier Algorithm to predict")
    st.caption("Evaluation is just a model understanding metrics using test data for your reference")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the model: {accuracy}")

    classification_rep = classification_report(y_test, y_pred)
    st.write("Classification Report:")
    st.write(classification_rep)