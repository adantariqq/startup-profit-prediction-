import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load the pre-trained model
model = joblib.load('startup_profit_model.pkl')

# Set page configuration
st.set_page_config(page_title="Startup Profit Predictor", layout="wide")

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
    }
    .stSubheader {
        color: #34495e;
        font-size: 1.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    rd_spend = st.number_input("R&D Spend ($)", min_value=0.0, step=1.0, value=10000.0)
    admin = st.number_input("Administration Cost ($)", min_value=0.0, step=1.0, value=50000.0)
    marketing = st.number_input("Marketing Spend ($)", min_value=0.0, step=1.0, value=20000.0)
    state = st.selectbox("Select State", ["California", "Florida", "New York"])

# Encode state variables
state_florida = 1 if state == "Florida" else 0
state_new_york = 1 if state == "New York" else 0

# Prediction history storage
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Main content area
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="stHeader">Startup Profit Predictor</h1>', unsafe_allow_html=True)
st.markdown("This tool predicts startup profits using a machine learning model based on your inputs.")

# Prediction button
if st.button("Generate Profit Prediction"):
    input_data = pd.DataFrame({
        'R&D Spend': [rd_spend],
        'Administration': [admin],
        'Marketing Spend': [marketing],
        'State_Florida': [state_florida],
        'State_New York': [state_new_york]
    })
    
    prediction = model.predict(input_data)[0]
    current_prediction = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "R&D Spend": rd_spend,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State": state,
        "Predicted Profit": prediction
    }
    st.session_state.predictions.append(current_prediction)
    
    st.success(f"Predicted Profit: ${prediction:.2f}")

# Display prediction history
st.markdown('<h2 class="stSubheader">Prediction History</h2>', unsafe_allow_html=True)
if st.session_state.predictions:
    prediction_df = pd.DataFrame(st.session_state.predictions)
    st.table(prediction_df)
else:
    st.info("No predictions made yet. Please generate a prediction.")

# Footer
st.markdown("---")
st.markdown('<span style="color: #34495e; font-size: 0.9em;">Developed by Adan Tariq | Powered by Streamlit | Last Updated: September 28, 2025, 12:18 PM PKT</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)