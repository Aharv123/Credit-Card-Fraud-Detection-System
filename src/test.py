import pandas as pd
import joblib
import numpy as np

# Function to load the trained model and the scaler
def load_model_and_scaler(model_path='D:/Credit Card Fraud Detection Project/models/best_model.pkl', 
                          scaler_path='D:/Credit Card Fraud Detection Project/models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to make predictions using the model and scaler
def predict_transaction(transaction_data, model, scaler):
    # Drop 'Class' column if present, as it is not used in prediction
    X = transaction_data.drop(columns=['Class'], errors='ignore')
    X_scaled = scaler.transform(X)

    # Predict the class and the probability for fraud detection
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1]

    return prediction, probability

# Function to get user input and predict the transaction result
def get_user_input_and_predict():
    print("Please enter the transaction details as comma-separated values (features are numerical):")
    print("Example: Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount")
    
    try:
        # Accept user input as a single line of comma-separated values
        user_input = input("\nEnter values: ")

        # Convert the input string into a list of features
        input_values = [float(i) for i in user_input.split(',')]

        # Ensure there are exactly 30 features
        if len(input_values) != 30:
            print("Invalid input! Please provide exactly 30 values.")
            return

        # Create a dictionary to match the feature names with the input values
        features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        # Map the input values to the feature names
        transaction_data = dict(zip(features, input_values))

        # Convert the dictionary to a DataFrame
        transaction_df = pd.DataFrame(transaction_data, index=[0])

        # Load model and scaler
        model, scaler = load_model_and_scaler()

        # Make predictions
        prediction, probability = predict_transaction(transaction_df, model, scaler)

        # Set the threshold for fraud detection
        THRESHOLD = 0.05  # You can adjust this based on your needs
        if probability[0] >= THRESHOLD:
            print(f"\nFraud detected! Probability: {probability[0]:.4f}")
        else:
            print(f"\nLegitimate transaction. Probability: {probability[0]:.4f}")
    
    except ValueError:
        print("Invalid input! Please ensure you entered valid numerical values.")

# Main function to run the script
if __name__ == "__main__":
    while True:
        get_user_input_and_predict()
        again = input("\nDo you want to check another transaction? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting...")
            break
