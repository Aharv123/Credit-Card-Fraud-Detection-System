This project aims to detect fraudulent credit card transactions using machine learning techniques. By analyzing a dataset of credit card transactions, it builds and trains a machine learning model to classify transactions as either legitimate or fraudulent.


"Folder: data"
   -> Consist of creditcard.csv   


"Folder:  Jupyter Notebook"
   -> In this i have created ipynb files in order to Daata Cleaning, Data Preprocessing, Data Visualization, checking data imbalance etc (data_exploration.ipynb)
   -> Using ML algorithms Logistic Regression, Random Forest, XGBoost to train the dataset, and evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.(training.ipynb)


"Folder: models "
   -> This folder is where your trained machine learning assets are stored after running the training scripts. These files are essential for making predictions without retraining the model every time. Instead of training the model again from scratch each time, you load this .pkl file and use it to make predictions.


"Folder: src"
   -> This folder includes scripts and modules that manage everything from preprocessing data to training the model and making predictions.

        data_preprocessing.py -   Prepares and scales data	
        model.py	          -   Trains and saves the best model	
        predictor.py	      -   Makes predictions with saved model	
        utils.py	          -   Helper functions for reuse	



"STEPS"


1️⃣ Download the Dataset
Download the dataset from this Kaggle link:
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


2️⃣ Install Required Libraries
In your terminal, run the following command to install all dependencies:

pip install -r requirements.txt


3️⃣ Create a Folder for the Dataset
Create a folder named data in the root directory of the project, and place the downloaded creditcard.csv file inside it.


4️⃣ Also Add the Dataset to Jupyter Folder
Place a copy of the same CSV file directly into the Jupyter Notebook folder as well.

⚠️ This is required because the CSV file was too large to push to GitHub, so it's ignored using .gitignore.

5️⃣ Run Data Exploration Notebook
Open and run the notebook Jupyter Notebook/data_exploration.ipynb to analyze and visualize the dataset.


6️⃣ Run Model Training Notebook
Next, run Jupyter Notebook/training.ipynb to train and evaluate different models.


7️⃣ Execute Source Scripts
Now move to the src/ directory and run the scripts in this order:

1 data_preprocessing.py
2 model.py

✅ Finally, run "test.py"
This is the main file to make predictions on whether a credit card transaction is fraudulent or not.



"" AND YEAH YOU ARE GOOD TO GO WITH THE PROJECT !! HOPE YOU LIKE IT ""

