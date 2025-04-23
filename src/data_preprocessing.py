import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path='data/creditcard.csv'):
    # Load the dataset
    df = pd.read_csv(path)

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale the 'Time' and 'Amount' columns
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Split features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y, scaler

# Optional: Run this script standalone to test
if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data()
    print("✅ Data loaded and preprocessed successfully.")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
