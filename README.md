💳 Credit Card Fraud Detection System
This project aims to detect fraudulent credit card transactions using machine learning techniques. By analyzing a dataset of credit card transactions, it builds and trains a machine learning model to classify transactions as either legitimate or fraudulent.

📽️ Demo
🔴 Real-time prediction for fraud detection
🟢 Model identifies whether a transaction is legitimate or fraudulent
💥 Fraudulent transactions are flagged, with predictions and probability displayed

🧠 Tech Stack

Technology	Purpose
Python	Main programming language
scikit-learn	Machine learning algorithms for classification
XGBoost	Advanced model training for better accuracy
Pandas	Data cleaning & manipulation
NumPy	Data processing and mathematical operations
Matplotlib/Seaborn	Data visualization for EDA and analysis
Flask	For creating a simple web app for predictions
📂 File Structure
The project is organized as follows:

bash
Copy
Edit
/Credit-Card-Fraud-Detection-System
    /data
        creditcard.csv
    /Jupyter Notebook
        data_exploration.ipynb
        training.ipynb
    /src
        data_preprocessing.py
        model.py
        predictor.py
    /models
        best_model.pkl
        scaler.pkl
    /requirements.txt
    /README.md
/src: Contains Python scripts for data preprocessing, model training, and prediction.

/Jupyter Notebook: Jupyter notebooks for data exploration and training the model.

/models: Saved models and scaler used for prediction.

/data: Folder to store the dataset.

🛠️ Installation
1. Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system
2. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
3. Download the Dataset:
Download the Credit Card Fraud Dataset from Kaggle:

Credit Card Fraud Dataset on Kaggle

4. Dataset Setup:
Create a folder named data in the root directory of the project.

Place the downloaded creditcard.csv file inside the data folder.

Also, place the creditcard.csv file directly inside the Jupyter Notebook folder for convenience.

🚀 Step-by-Step Instructions
1. Data Preprocessing:
Run the data_preprocessing.py script first to clean and preprocess the data. This step handles tasks like feature extraction, normalization, and preparing the data for training.

bash
Copy
Edit
python src/data_preprocessing.py
2. Model Training:
Next, run the model.py script to build and train the machine learning model using the preprocessed data.

bash
Copy
Edit
python src/model.py
3. Prediction:
After training the model, use the predictor.py script to make predictions on new or test data.

bash
Copy
Edit
python src/predictor.py
💻 Requirements
To set up the environment and install dependencies, run the following:

bash
Copy
Edit
pip install -r requirements.txt
The requirements.txt file includes all the necessary libraries for the project.
