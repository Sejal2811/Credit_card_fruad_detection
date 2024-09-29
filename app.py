from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

app = Flask(__name__)

# Re-load the model
loaded_model = joblib.load('logistic_model.pkl')

@app.route('/')
def home():
    # Plot graphs (e.g., distribution of fraudulent transactions)
    df = pd.read_csv('preprocessFraudTrain.csv')

    plt.figure(figsize=(7.5,6))
    plt.hist(df['age'], bins=50, color='lightgreen', edgecolor='black')
    plt.title('Age Distribution', fontsize=16)
    plt.xlabel('age', fontsize=12)
    plt.ylabel('is_fraud', fontsize=12)  # Y-axis should show frequency instead of is_fraud
    plt.grid(True)
    plt.title('Age Distribution ')
    # Save the plot as an image in the static directory
    plt.savefig('static/age_distribution.png')  # Save it in the 'static' folder
    plt.close() # Plot histogram for age
    plt.figure(figsize=(7.5,6))
    plt.hist(df['amt'], bins=50, color='lightgreen', edgecolor='black')
    plt.title('Amount Distribution', fontsize=16)
    plt.xlabel('amt', fontsize=12)
    plt.ylabel('is_fraud', fontsize=12)
    plt.grid(True)
    plt.title('Transaction Amount Distribution')
    plt.savefig('static/amount_distribution.png')
    plt.close()
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Collect the form data from the prediction page
        try:
            cc_num = request.form['cc_num']
            category = request.form['category']
            amt = int(request.form['amt'])
            gender = int(request.form['gender'])
            trans_num = float(request.form['trans_num'])
            age = int(request.form['age'])

            # Prepare input data for the model
            input_data = {
                'cc_num': [cc_num],
                'category': [category],
                'amt': [amt],
                'gender': [gender],
                'trans_num': [trans_num],
                'age': [age]
            }
            
            input_df = pd.DataFrame(input_data)
            input_df['cc_num'] = pd.to_numeric(input_df['cc_num'], errors='coerce')

            print(loaded_model)
            print(input_df)
            print(input_df.dtypes)

            # Make prediction using the loaded model
            prediction = loaded_model.predict(input_df)

            # Render the result on the prediction page
            return render_template('predict.html', result=prediction[0])
        except Exception as e:
            # Handle any error that might occur and return a message
            error_message = f"An error occurred: {e}"
            print(error_message)  # You can also log this in a real-world app
            return render_template('predict.html', result=error_message)

    # Handle GET requests
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
