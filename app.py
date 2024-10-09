import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os


# Load the trained model
model_file = 'logistic_model.pkl'  # Make sure this is the correct path to your model
fraud_model = joblib.load(model_file)

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Homepage", "Predict"])

# Homepage
if page == "Homepage":
    st.title("Credit Card Fraud Analyzer")
    st.write("Welcome to the credit card fraud detection system.")
    st.write("Use this app to check whether a transaction is fraudulent or not.")
    st.write("""
        ### Instructions:
        - Go to the **Predict** page to enter transaction details.
        - The model will analyze the transaction and predict if it's fraudulent or not.
    """)
    
    # Image Slider Section
    st.write(" Plotting the graph ")
    
    # List of image paths
    images = [
        ('static/age_distribution.png', "Age Distribution of Transactions"),
        ('static/amount_distribution.png', "Amount Distribution of Transactions"),
        ('static/Category.png', "Transaction Categories Overview")
    ]

    # Create a horizontal layout for images
    cols = st.columns(len(images))  # Create columns based on the number of images
    for col, (image_path, description) in zip(cols, images):
        # Check if the image file exists before attempting to open it
        if os.path.exists(image_path):
            image = Image.open(image_path)
            col.image(image, caption=description, use_column_width='always')  # Set image width to always fit the column
        else:
            col.warning(f"Image not found: {image_path}")

# Predict Page
elif page == "Predict":
    st.title("Fraud Prediction")
    st.write("Input transaction details to check for potential fraud.")

    # Create a form for user input
    with st.form("fraud_detection_form"):
        cc_num = st.text_input("Credit Card Number", "")
        category = st.selectbox("Transaction Category", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
        gender = st.selectbox("Gender", ['0 (Male)', '1 (Female)'])
        trans_num = st.text_input("Transaction Number", "")
        age = st.number_input("Age", min_value=0, step=1)

        # Submit button
        submit = st.form_submit_button("Check for Fraud")

    # Handle form submission
    if submit:
        try:
            # Prepare the input for prediction
            input_data = pd.DataFrame({
                'cc_num': [int(cc_num)],
                'category': [category],
                'amt': [amt],
                'gender': [int(gender[0])],  # Extract the integer value from the gender string
                'trans_num': [float(trans_num)],
                'age': [age]
            })

            # Make prediction
            prediction = fraud_model.predict(input_data)[0]

            # Display the result
            if prediction == 1:
                st.error("Warning: This transaction is predicted as **Fraudulent**!")
            else:
                st.success("This transaction is predicted as **Non-Fraudulent**.")

        except ValueError as e:
            st.error(f"An error occurred: {e}")
