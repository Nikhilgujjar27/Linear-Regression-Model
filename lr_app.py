import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the trained model (ensure the file path is correct)
try:
    model = pickle.load(open('linear_regression_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please check the path.")
    st.stop()

# Create a webpage
st.title('Scikit-learn Linear Regression Model')

# Input fields for user data
tv = st.text_input("Enter TV Advertising Spend")
radio = st.text_input("Enter Radio Advertising Spend")
newspaper = st.text_input("Enter Newspaper Advertising Spend")

# Prediction button
if st.button("Predict"):
    try:
        # Convert the input to float and create the features array
        tv = float(tv) if tv else 0.0  # Default to 0 if empty
        radio = float(radio) if radio else 0.0
        newspaper = float(newspaper) if newspaper else 0.0

        # Ensure the model is expecting three inputs
        features = np.array([[tv, radio, newspaper]], dtype=np.float32)

        # Make prediction
        result = model.predict(features)

        # Show the result
        st.write("Predicted results: ", result[0])
    except ValueError:
        st.error("Please enter valid numeric values for TV, Radio, and Newspaper.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
