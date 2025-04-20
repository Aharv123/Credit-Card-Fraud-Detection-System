import pandas as pd
import joblib
import numpy as np

# Load model and scaler
def load_model_and_scaler(model_path='../models/best_model.pkl', scaler_path='../models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Predict transaction
def predict_transaction(transaction_data, model, scaler):
    # Ensure the dataframe columns match those used during training (e.g., Time, Amount, V1, V2, ..., V28)
    # We assume 'transaction_data' is a 1-row DataFrame with the same columns as training data

    # Separate features (excluding the target 'Class' column)
    X = transaction_data.drop(columns=['Class'], errors='ignore')

    # Scale all numeric columns (assuming scaler was fitted on all features except 'Class')
    X_scaled = scaler.transform(X)

    # Predict using the model
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1]

    return prediction, probability

# Function to interactively get user input and predict the transaction
def get_user_input_and_predict():
    # Get user input for the transaction details
    print("Please enter the transaction details (features are numerical):\n")
    try:
        transaction_data = {
            'Time': [float(input("Enter Time: "))],
            'V1': [float(input("Enter V1: "))],
            'V2': [float(input("Enter V2: "))],
            'V3': [float(input("Enter V3: "))],
            'V4': [float(input("Enter V4: "))],
            'V5': [float(input("Enter V5: "))],
            'V6': [float(input("Enter V6: "))],
            'V7': [float(input("Enter V7: "))],
            'V8': [float(input("Enter V8: "))],
            'V9': [float(input("Enter V9: "))],
            'V10': [float(input("Enter V10: "))],
            'V11': [float(input("Enter V11: "))],
            'V12': [float(input("Enter V12: "))],
            'V13': [float(input("Enter V13: "))],
            'V14': [float(input("Enter V14: "))],
            'V15': [float(input("Enter V15: "))],
            'V16': [float(input("Enter V16: "))],
            'V17': [float(input("Enter V17: "))],
            'V18': [float(input("Enter V18: "))],
            'V19': [float(input("Enter V19: "))],
            'V20': [float(input("Enter V20: "))],
            'V21': [float(input("Enter V21: "))],
            'V22': [float(input("Enter V22: "))],
            'V23': [float(input("Enter V23: "))],
            'V24': [float(input("Enter V24: "))],
            'V25': [float(input("Enter V25: "))],
            'V26': [float(input("Enter V26: "))],
            'V27': [float(input("Enter V27: "))],
            'V28': [float(input("Enter V28: "))],
            'Amount': [float(input("Enter Amount: "))]
        }
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Convert input into a DataFrame
    transaction_df = pd.DataFrame(transaction_data)

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    # Make prediction
    prediction, probability = predict_transaction(transaction_df, model, scaler)

    # Output result
    if prediction == 1:
        print(f"Fraud detected! Probability: {probability[0]:.4f}")
    else:
        print(f"Legitimate transaction. Probability: {probability[0]:.4f}")

if __name__ == "__main__":
    # Call the function to get user input and predict the transaction
    get_user_input_and_predict()
