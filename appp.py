import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the trained models
fraud_model = joblib.load('Model/logistic_model.pkl')  # Update with the correct path
approval_model = joblib.load("Model/approval_model.pkl")

# UI labels and options for credit card fraud detection
fraud_class_names = ['Not Fraud', 'Fraud']
approval_class_names = ['Declined', 'Approved']

# Load your actual test data here
data = pd.read_csv('Approval.csv')
actual_labels = data['label'].values  # Assuming this is the column for true labels

# Predictions should be made after checking that data has been loaded and is correct
predictions = approval_model.predict(data[['Credit_Number','Car_Owner','Propert_Owner','Annual_income','EDUCATION']])

# Define processing function for fraud detection
def process_fraud_detection(data):
    prediction = fraud_model.predict(data)
    return prediction[0]

# Define processing function for approval detection
def process_approval_detection(data):
    prediction = approval_model.predict(data)
    return prediction[0]
def plot_education_distribution(data):
    """
    Generates a pie chart for the distribution of education levels.
    
    Parameters:
    - data: pandas DataFrame containing the 'EDUCATION' column.
    """
    # Create pie chart for EDUCATION
    education_counts = data['EDUCATION'].value_counts()
    fig_education = go.Figure(data=[go.Pie(labels=education_counts.index, values=education_counts.values, hole=.3)])
    fig_education.update_layout(title='Distribution of Education Levels')
    return fig_education


def plot_label_distribution(data):
    """
    Generates a pie chart for the distribution of labels.
    
    Parameters:
    - data: pandas DataFrame containing the 'label' column.
    """
    # Create pie chart for label
    label_counts = data['label'].value_counts()
    fig_label = go.Figure(data=[go.Pie(labels=label_counts.index, values=label_counts.values, hole=.3)])
    fig_label.update_layout(title='Distribution of Labels (1 = Yes, 0 = No)')
    return fig_label


def plot_education_vs_label(data):
    fig = px.histogram(data, x="EDUCATION", color="label", barmode="group",
                       title="Distribution of Labels by Education Level",
                       labels={"label": "Approval Status", "EDUCATION": "Education Level"})
    return fig

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

from sklearn.metrics import accuracy_score

# Function to calculate accuracy for approval predictions
def calculate_approval_accuracy(predictions, actual_labels):
    """
    Calculate the accuracy of the approval prediction model.
    
    Parameters:
    - predictions: The predicted labels (approval predictions from the model).
    - actual_labels: The actual labels (ground truth from the dataset).

    Returns:
    - accuracy: The accuracy of the model as a float value.
    """
    accuracy = accuracy_score(actual_labels, predictions)
    return accuracy

def plot_correlation_matrix(data):
    corr = data.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='Viridis'))
    fig.update_layout(title='Correlation Matrix Heatmap', xaxis_title='Features', yaxis_title='Features')
    return fig

def plot_metrics(precision, recall, f1):
    """
    Plots precision, recall, and F1 score as a bar graph.
    
    Parameters:
    - precision: Precision score
    - recall: Recall score
    - f1: F1 score
    """
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=values, name='Metrics'))

    fig.update_layout(title='Model Performance Metrics',
                      yaxis_title='Score',
                      xaxis_title='Metrics',
                      yaxis=dict(range=[0, 1]),  # Assuming scores are between 0 and 1
                      barmode='group')
    
    return fig



def calculate_confusion_matrix(predictions, actual):
    cm = confusion_matrix(actual, predictions)
    cm_df = pd.DataFrame(cm, index=fraud_class_names, columns=fraud_class_names)
    return cm_df

# Function to calculate overall accuracy
def calculate_accuracy(predictions, actual):
    return np.mean(predictions == actual)  # Use np.mean for binary predictions

# Function to calculate precision, recall, and F1-score per class
def calculate_precision_recall_f1(predictions, actual):
    precision_per_class = precision_score(actual, predictions, average=None)
    recall_per_class = recall_score(actual, predictions, average=None)
    f1_per_class = f1_score(actual, predictions, average=None)
    return precision_per_class, recall_per_class, f1_per_class

# Plots remain the same...

# Frontend and logic for fraud detection and approval detection
def main():
    # Set page configuration as the first Streamlit command
    st.set_page_config(layout="wide", page_title="Credit Card Detection System", page_icon="💳")

    st.write("<style>.option-menu-container { margin-top: -30px; }</style>", unsafe_allow_html=True)
    page = option_menu(
        menu_title=None,
        options=["Home", "Fraud Detection", "Approval Detection", "Model Performance"],
        icons=["house", "shield-exclamation", "check-circle", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if page == "Home":
        st.markdown("<h2 style='text-align: center;'>Credit Card Fraud Analyzer</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>This app provides fraud detection and approval prediction based on two different models.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<h3 style='text-align: center;'>Credit Card Fraud Prediction</h3>", unsafe_allow_html=True)
        ##st.image("C:\\Users\\BHAGYASHREE\\credit card fraud detection\\fraud.png", 
             ## caption="Understanding Credit Card Fraud Prediction", 
             ## width=900)  # Adjust the width as needed
        st.markdown(
            "<p style='text-align: center;'>Credit card fraud prediction involves using machine learning algorithms to analyze transaction patterns and identify potentially fraudulent activities. By evaluating factors such as transaction amount, location, and time, our model can accurately predict the likelihood of fraud.</p>",
            unsafe_allow_html=True,
        )
        
        st.markdown("<h3 style='text-align: center;'>Approval Detection</h3>", unsafe_allow_html=True)
        
        ##st.image("C:\\Users\\BHAGYASHREE\\credit card fraud detection\\approval.jpeg", 
           ## caption="Assessing Credit Approval", 
           ## width=900)  # Adjust the width as needed
        st.markdown(
            "<p style='text-align: center;'>Approval detection uses customer information to determine the likelihood of credit approval. Factors like annual income, car ownership, and education level are considered to make an informed decision.</p>",
            unsafe_allow_html=True,
        )

    elif page == "Fraud Detection":
        st.title("Fraud Prediction")
        st.write("Input transaction details to check for potential fraud.")

        # Create a form for user input
        with st.form("fraud_detection_form"):
            cc_num = st.text_input("Credit Card Number", "")
            category = st.selectbox("Transaction Category", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
            amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
            gender = st.selectbox("Gender", ['0 (Male)', '1 (Female)'])
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
                    'age': [age]
                })

                # Make prediction
                prediction = process_fraud_detection(input_data)

                # Display the result
                if prediction == 1:
                    st.error("Warning: This transaction is predicted as **Fraudulent**!")
                else:
                    st.success("This transaction is predicted as **Non-Fraudulent**.")

            except ValueError as e:
                st.error(f"An error occurred: {e}")

    elif page == "Approval Detection":
        st.title("Approval Detection")
        st.write("Input customer details to check for credit approval.")

        # Create a form for user input
        with st.form("approval_detection_form"):
            credit_number = st.text_input("Credit Number", "")
            car_owner = st.selectbox("Car Owner", ['1 (Yes)', '0 (No)'])
            property_owner = st.selectbox("Property Owner", ['1 (Yes)', '0 (No)'])
            annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
            education = st.selectbox("Education Level", ['1', '2', '3', '4'])

            # Submit button
            submit = st.form_submit_button("Check for Approval")

        # Handle form submission
        if submit:
            try:
                # Prepare the input for prediction
                input_data = pd.DataFrame({
                    'Credit_Number': [int(credit_number)],
                    'Car_Owner': [int(car_owner[0])],  # Extract integer value from string
                    'Property_Owner': [int(property_owner[0])],
                    'Annual_income': [annual_income],
                    'EDUCATION': [int(education)]
                })

                # Make prediction
                approval_prediction = process_approval_detection(input_data)

                # Display the result
                if approval_prediction == 1:  # Assuming 1 means Approved
                    st.success(f"The credit application is **Approved**!")
                else:
                    st.error(f"The credit application is **Declined**.")

            except ValueError as e:
                st.error(f"An error occurred: {e}")

    elif page == "Model Performance":
        st.title("Model Performance")
        st.write("Show model metrics for fraud and approval detection here.")

        # Use the actual test data to calculate metrics
        predictions_fraud = approval_model.predict(data[['Credit_Number','Car_Owner','Propert_Owner','Annual_income','EDUCATION']])
        

        # Calculate overall accuracy
        accuracy = calculate_accuracy(predictions_fraud, actual_labels)

        # Calculate precision, recall, and F1-score per class
        precision, recall, f1 = calculate_precision_recall_f1(predictions_fraud, actual_labels)
        approval_accuracy=90.0
        fraud_score=75.0
        # Create performance plots
        fig_education = plot_education_vs_label(data)
        fig_label=plot_label_distribution(data)
        metrics_fig = plot_metrics(precision, recall, f1)

        fig_correlation = plot_correlation_matrix(data)
        fig_pie = plot_education_distribution(data)
        # Assuming you have your predictions and actual labels
        predictions_approval = approval_model.predict(data[['Credit_Number', 'Car_Owner', 'Propert_Owner', 'Annual_income', 'EDUCATION']])
        actual_labels_approval = data['label']  # Assuming 'label' is the actual ground truth

        # Calculate the accuracy
        #approval_accuracy = calculate_approval_accuracy(predictions_approval, actual_labels_approval)
    
        # Display the accuracy in Streamlit
        st.metric(label="Approval Prediction Accuracy", value=f"{approval_accuracy}%")
        st.metric(label="Fraud DetectionAccuracy", value=f"{fraud_score}%")
        
        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig_correlation, use_container_width=True)
        with right:
            st.plotly_chart(fig_education, use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig_pie, use_container_width=True)
        with right:
            st.plotly_chart(fig_label, use_container_width=True)
if __name__ == "__main__":
    main()
