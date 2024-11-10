import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("binary_model.h5")

def get_user_input():
    st.sidebar.header("Enter Patient Data")

    age = st.sidebar.slider("Age", 1, 120, 30)
    sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.sidebar.slider("Chest Pain Type (0 to 3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.sidebar.slider("Resting Electrocardiographic Results (0 to 2)", 0, 2, 1)
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
    slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (0 to 2)", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy (0 to 3)", 0, 3, 0)
    thal = st.sidebar.slider("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", 1, 3, 2)

    user_data = {
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
    }

    return np.array([list(user_data.values())])

def main():
    st.title("Heart Disease Prediction")

    user_input = get_user_input()

    if st.button("Predict"):
        prediction = model.predict(user_input)
        if prediction[0][0] > 0.5:
            st.write("**Prediction:** The model predicts that the patient has heart disease.")
        else:
            st.write("**Prediction:** The model predicts that the patient does not have heart disease.")

if __name__ == "__main__":
    main()
