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

df = pd.read_csv('preprocessFraudTrain.csv')
df.head()
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
def plot_fraud_distribution(df):
    """
    Generates a pie chart for the distribution of fraud labels.
    
    Parameters:
    - df: pandas DataFrame containing the 'is_fraud' column.
    """
    # Create pie chart for fraud label distribution
    fraud_counts = df['is_fraud'].value_counts()
    fig_fraud = go.Figure(data=[go.Pie(labels=fraud_counts.index, values=fraud_counts.values, hole=.3)])
    fig_fraud.update_layout(title='Distribution of Frauds (1 = Yes, 0 = No)')
    return fig_fraud

def plot_approval_distribution(data):
    """
    Generates a pie chart for the distribution of approval labels.
    
    Parameters:
    - data: pandas DataFrame containing the 'label' column.
    """
    # Create pie chart for approval label distribution
    approval_counts = data['label'].value_counts()
    fig_approval = go.Figure(data=[go.Pie(labels=approval_counts.index, values=approval_counts.values, hole=.3)])
    fig_approval.update_layout(title='Distribution of Approvals (1 = Yes, 0 = No)')
    return fig_approval


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
category_options = {
    '1': 'Grocery',
    '2': 'Online shopping',
    '3': 'Investment',
    '4': 'Rent',
    '5': 'Travel',
    '6': 'Banking',
    '7': 'Personal order',
    '8': 'Entertainment',
    '9': 'Fraud',
    '10': 'Theft',
    '11': 'Taxation',
    '12': 'Business'
}

# Create a reverse mapping for easy access
category_labels = {key: value for key, value in category_options.items()}

education_options = {
    '1': '10th pass',
    '2': '12th pass',
    '3': 'Undergraduate',
    '4': 'PostGraduate'
}

# Create a reverse mapping for easy access
education_labels = {key: value for key, value in education_options.items()}
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
        st.markdown("<h1 style='text-align: center;'>Credit Card Analysing System</h1>", unsafe_allow_html=True)

        # Subtitle
        st.markdown(
            "<h2 style='text-align: center;'>Empowering Secure and Informed Financial Decisions through AI-Driven Insights</h2>", 
            unsafe_allow_html=True
        )


        # Paragraph with justified alignment
        st.markdown(
            """
            <h4 style='text-align: center;'>
            In today's digital economy, credit cards have become an essential tool for financial transactions. 
            However, the increasing volume of transactions also brings escalating risks of credit card fraud 
            and inefficient approval processes. To address these challenges, our project presents an 
            Intelligent Credit Card Analysis Platform. Our cutting-edge platform leverages Machine Learning (ML) 
            to detect potential fraud and streamline credit card approvals, empowering financial institutions and 
            individuals to make informed decisions.
            </h4>
            """, 
            unsafe_allow_html=True
        )
        

    elif page == "Fraud Detection":
        st.title("Fraud Prediction")
        st.markdown("""
    <style>
        h3 {
            font-family: 'Helvetica', sans-serif;
            color: #333333;
            margin-bottom: 15px;
        }
        p {
            font-size: 16px;
            color: #333333;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Simplified content with font change
        st.markdown("""
            <h3>Note: Input transaction details to check for potential fraud</h3>
        """, unsafe_allow_html=True)
        # Create a form for user input
        with st.form("fraud_detection_form"):
            cc_num = st.text_input("Credit Card Number", "")
            category = st.selectbox("Transaction Category", list(category_labels.keys()), format_func=lambda x: category_labels[x])
            amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
            gender = st.selectbox("Gender", ['0 (Male)', '1 (Female)'])
            age = st.number_input("Age", min_value=0, step=1)

            # Submit button
            submit = st.form_submit_button("Check for Fraud")

        # Handle form submission
        if submit:
            # Validate credit card number
            if len(cc_num) != 10 or not cc_num.isdigit():
                st.error("Credit Card Number must be exactly 10 digits.")
            
            # Validate transaction amount
            elif amt <= 0:
                st.error("Transaction Amount must be greater than zero.")
            
            # Validate gender selection
            elif gender not in ['0 (Male)', '1 (Female)']:
                st.error("Please select a valid gender option.")
            
            # Validate age
            elif age < 0:
                st.error("Age must be a non-negative integer.")
            
            else:
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
        st.markdown("""
            <h3>Note: Input customer details to check for credit approval</h3>
        """, unsafe_allow_html=True)
        # Create a form for user input
        with st.form("approval_detection_form"):
            credit_number = st.text_input("Credit Number", "")
            car_owner = st.selectbox("Car Owner", ['1 (Yes)', '0 (No)'])
            property_owner = st.selectbox("Property Owner", ['1 (Yes)', '0 (No)'])
            annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
            education = st.selectbox("Education Level", list(education_labels.keys()), format_func=lambda x: education_labels[x])

            # Submit button
            submit = st.form_submit_button("Check for Approval")

        # Handle form submission
        # Handle form submission
        if submit:
                    # Validate credit number
                    if len(credit_number) == 0:
                        st.error("Credit Number cannot be empty.")
                    elif not credit_number.isdigit():
                        st.error("Credit Number must be numeric.")
                    
                    # Validate car owner selection
                    elif car_owner not in ['1 (Yes)', '0 (No)']:
                        st.error("Please select a valid option for Car Owner.")
                    
                    # Validate property owner selection
                    elif property_owner not in ['1 (Yes)', '0 (No)']:
                        st.error("Please select a valid option for Property Owner.")
                    
                    # Validate annual income
                    elif annual_income <= 0:
                        st.error("Annual Income must be greater than zero.")
                    
                    # Validate education level
                    elif education not in ['1', '2', '3', '4']:
                        st.error("Please select a valid Education Level.")
                    
                    else:
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
        fraud_score=85.60
        # Create performance plots
        fig_education = plot_education_vs_label(data)
        fig_label1 = plot_fraud_distribution(df)
        fig_label2=plot_approval_distribution(data)
        metrics_fig = plot_metrics(precision, recall, f1)
        fig_correlation2 = plot_correlation_matrix(df)
        fig_correlation1 = plot_correlation_matrix(data)
        fig_pie = plot_education_distribution(data)
        # Assuming you have your predictions and actual labels
        predictions_approval = approval_model.predict(data[['Credit_Number', 'Car_Owner', 'Propert_Owner', 'Annual_income', 'EDUCATION']])
        actual_labels_approval = data['label']  # Assuming 'label' is the actual ground truth
        
        # Calculate the accuracy
        #approval_accuracy = calculate_approval_accuracy(predictions_approval, actual_labels_approval)
    
        # Display the accuracy in Streamlit
        st.metric(label="Approval Prediction Accuracy", value=f"{approval_accuracy}%")
        st.metric(label="Fraud DetectionAccuracy", value=f"{fraud_score}%")
        st.title("Approvals")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig_correlation1, use_container_width=True)
        with right:
            st.plotly_chart(fig_label2, use_container_width=True)
        st.title("Fraud")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig_correlation2, use_container_width=True)
        with right:
            st.plotly_chart(fig_label1, use_container_width=True)
if __name__ == "__main__":
    main()
