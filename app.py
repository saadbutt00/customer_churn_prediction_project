import os
import pickle
import streamlit as st
import pandas as pd

# Define the absolute path to your files
absolute_path = 'F:/MyProject/Lib/customer churn prediction'

# Define file paths
model_path = os.path.join(absolute_path, 'rf_model.pkl')
encoder_path = os.path.join(absolute_path, 'label_encoders.pkl')
features_path = os.path.join(absolute_path, 'features.pkl')

# Load the trained model
try:
    with open(model_path, 'rb') as model_file:
        rf_model = pickle.load(model_file)
except FileNotFoundError as e:
    st.error(f"Error: {e}. The model file was not found at {model_path}.")
    st.stop()

# Load the label encoders
try:
    with open(encoder_path, 'rb') as encoder_file:
        label_encoders = pickle.load(encoder_file)
except FileNotFoundError as e:
    st.error(f"Error: {e}. The label encoders file was not found at {encoder_path}.")
    st.stop()

# Load the feature names
try:
    with open(features_path, 'rb') as feature_file:
        feature_names = pickle.load(feature_file)
except FileNotFoundError as e:
    st.error(f"Error: {e}. The features file was not found at {features_path}.")
    st.stop()

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields
def user_input_features():
    return pd.DataFrame({
        'gender': [st.sidebar.selectbox('Gender', options=['Male', 'Female'])],
        'Partner': [st.sidebar.selectbox('Partner', options=['Yes', 'No'])],
        'Dependents': [st.sidebar.selectbox('Dependents', options=['Yes', 'No'])],
        'tenure': [st.sidebar.slider('Tenure (Months)', 0, 72, 12)],
        'PhoneService': [st.sidebar.selectbox('Phone Service', options=['Yes', 'No'])],
        'MultipleLines': [st.sidebar.selectbox('Multiple Lines', options=['Yes', 'No'])],
        'OnlineSecurity': [st.sidebar.selectbox('Online Security', options=['Yes', 'No'])],
        'OnlineBackup': [st.sidebar.selectbox('Online Backup', options=['Yes', 'No'])],
        'DeviceProtection': [st.sidebar.selectbox('Device Protection', options=['Yes', 'No'])],
        'TechSupport': [st.sidebar.selectbox('Tech Support', options=['Yes', 'No'])],
        'StreamingTV': [st.sidebar.selectbox('Streaming TV', options=['Yes', 'No'])],
        'StreamingMovies': [st.sidebar.selectbox('Streaming Movies', options=['Yes', 'No'])],
        'Contract': [st.sidebar.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])],
        'PaperlessBilling': [st.sidebar.selectbox('Paperless Billing', options=['Yes', 'No'])],
        'PaymentMethod': [st.sidebar.selectbox('Payment Method', options=['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'])],
        'MonthlyCharges': [st.sidebar.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0)],
        'TotalCharges': [st.sidebar.number_input('Total Charges', min_value=0.0, max_value=8000.0, value=800.0)],
        'InternetService': [st.sidebar.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])],
        'SeniorCitizen': [st.sidebar.slider('Senior Citizen', 0, 1, 0)]
    })

# Get user input data
input_df = user_input_features()

# Encode user input using the label encoders
for column in input_df.columns:
    if column in label_encoders:
        input_df[column] = label_encoders[column].transform(input_df[column])

# Ensure input_df has the same columns as the model's training data
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Make prediction
new_churn_prediction = rf_model.predict(input_df)
prediction_proba = rf_model.predict_proba(input_df)

# Display results
st.write(f"**Prediction**: {'Churn' if new_churn_prediction[0] == 1 else 'No Churn'}")
st.write(f"**Prediction Probability**: Churn: {prediction_proba[0][1]:.2f}, No Churn: {prediction_proba[0][0]:.2f}")

# streamlit run "F:/MyProject/Lib/customer churn prediction/app.py"
