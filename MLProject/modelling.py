import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def run_retraining():
    # Ensure tracking directory exists
    if not os.path.exists('mlruns'):
        os.makedirs('mlruns')

    with mlflow.start_run():
        # Load and prepare training data
        raw_data = {
            'Income': [5400, 2100, 8100, 3100, 4800, 2500, 7200, 3900],
            'Loan_Amount': [150, 95, 280, 130, 190, 110, 240, 160],
            'Status': [1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(raw_data)
        
        X = df[['Income', 'Loan_Amount']]
        y = df['Status']
        
        # Initialize and train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Log metrics and model artifacts
        acc = model.score(X, y)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "loan_model")
        
        print(f"Retraining completed successfully. Accuracy: {acc}")

if __name__ == "__main__":
    run_retraining()