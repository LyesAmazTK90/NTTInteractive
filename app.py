import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Models
@st.cache
def load_models():
    with open('model/best_lasso.pkl', 'rb') as f:
        best_lasso = pickle.load(f)
    with open('model/best_gbm.pkl', 'rb') as f:
        best_gbm = pickle.load(f)
    with open('model/best_xgb.pkl', 'rb') as f:
        best_xgb = pickle.load(f)
    with open('model/stacking_model.pkl', 'rb') as f:
        stacking_model = pickle.load(f)
    return best_lasso, best_gbm, best_xgb, stacking_model

best_lasso, best_gbm, best_xgb, stacking_model = load_models()

# Load Data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

train_data = load_data('data/train.csv')
test_data = load_data('data/test.csv')

# Title
st.title("House Price Prediction")

# Sidebar for Input Features
st.sidebar.header("Input Features")
features = {
    'OverallQual': st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5),
    'GrLivArea': st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500),
    'GarageCars': st.sidebar.slider("Garage Cars Capacity", 0, 5, 2),
    'GarageArea': st.sidebar.number_input("Garage Area (sq ft)", 0, 1500, 500),
    # Add more features as needed...
}

# Convert inputs to DataFrame
input_df = pd.DataFrame([features])

# Predict
def predict(input_data):
    # Preprocess input as per training (e.g., scaling, encoding)
    # Example: scaled_data = scaler.transform(input_data)
    
    lasso_pred = best_lasso.predict(input_data)
    gbm_pred = best_gbm.predict(input_data)
    xgb_pred = best_xgb.predict(input_data)
    stacked_input = np.column_stack((lasso_pred, gbm_pred, xgb_pred))
    final_pred = stacking_model.predict(stacked_input)
    return np.exp(final_pred[0])

# Display prediction
st.subheader("Predicted Sale Price")
predicted_price = predict(input_df)
st.write(f"${predicted_price:,.2f}")

# Visualizations (e.g., correlation, feature importance)
st.subheader("Data Overview")
st.write(train_data.head())
st.bar_chart(train_data[['OverallQual', 'SalePrice']].corr())