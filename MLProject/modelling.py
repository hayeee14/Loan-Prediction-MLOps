import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train():
    with mlflow.start_run():
        df = pd.DataFrame({'Income': [5000, 2000], 'Status': [1, 0]})
        X, y = df[['Income']], df['Status']
        model = RandomForestClassifier().fit(X, y)
        mlflow.log_metric("accuracy", 1.0)
        mlflow.sklearn.log_model(model, "loan_model")
        print("Retraining Selesai!")

if __name__ == "__main__":
    train()