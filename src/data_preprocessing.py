import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    
    # Standard scaling on 'Time' and 'Amount'
    scaler = StandardScaler()
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y, scaler
