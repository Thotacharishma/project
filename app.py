import streamlit as st
import joblib
import numpy as np

# Load models and encoders
knn_model = joblib.load("knn_model.pkl")
decision_tree_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Income Prediction App")

def get_user_input():
    try:
        age = st.number_input("Age", 17, 90)
        workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
        education = st.selectbox("Education", label_encoders['education'].classes_)
        marital_status = st.selectbox("Marital Status", label_encoders['marital_status'].classes_)  # Updated key name
        occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
        relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
        race = st.selectbox("Race", label_encoders['race'].classes_)
        sex = st.selectbox("Sex", label_encoders['sex'].classes_)
        hours_per_week = st.number_input("Hours per Week", 1, 100)

        user_data = {
            'age': age,
            'workclass': label_encoders['workclass'].transform([workclass])[0],
            'education': label_encoders['education'].transform([education])[0],
            'marital-status': label_encoders['marital_status'].transform([marital_status])[0],  # Updated key name
            'occupation': label_encoders['occupation'].transform([occupation])[0],
            'relationship': label_encoders['relationship'].transform([relationship])[0],
            'race': label_encoders['race'].transform([race])[0],
            'sex': label_encoders['sex'].transform([sex])[0],
            'hours-per-week': hours_per_week
        }

        return np.array(list(user_data.values())).reshape(1, -1)
    
    except KeyError as e:
        st.error(f"KeyError: {e}. Please check if all the expected columns are present in the label encoders.")
        return None


user_input = get_user_input()

if st.button("Predict with KNN"):
    scaled_input = scaler.transform(user_input)
    result = knn_model.predict(scaled_input)
    st.write("Prediction (KNN):", "Income >50K" if result[0] == 1 else "Income <=50K")

if st.button("Predict with Decision Tree"):
    scaled_input = scaler.transform(user_input)
    result = decision_tree_model.predict(scaled_input)
    st.write("Prediction (Decision Tree):", "Income >50K" if result[0] == 1 else "Income <=50K")
