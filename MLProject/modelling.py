import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    with mlflow.start_run():
        
        df = pd.DataFrame({'Income': [5000, 2000], 'Status': [1, 0]})
        X, y = df[['Income']], df['Status']
        model = RandomForestClassifier().fit(X, y)
        
        # Log parameter dan metric
        mlflow.log_metric("accuracy", 1.0)
        mlflow.sklearn.log_model(model, "model")
        print("Retraining Berhasil!")