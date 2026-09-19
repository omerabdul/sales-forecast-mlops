import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# --- MLFLOW CONFIGURATION ---
# Dynamic URI to prevent 'Connection Refused' in GitHub Actions
if "public_url" in globals():
    mlflow.set_tracking_uri(public_url)
else:
    # Fallback for GitHub Runner
    tracking_uri = "file://" + os.path.abspath("mlruns")
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Sales-Forecast-Prediction-Experiment")

# CONFIGURATION
MODEL_REPO_ID = "flyingdragon98/salesforecast-model"
DATASET_REPO_ID = "flyingdragon98/salesforecast"
DATA_DIR = "mlops/data"
HF_TOKEN = os.getenv("HF_TOKEN")

# Check if there's an active run and end it if so
if mlflow.active_run():
    mlflow.end_run()

api = HfApi()

def train_with_grid_search():
    print("Step 1: Loading train/test datasets from Hugging Face...")

    Xtrain_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtrain.csv", repo_type="dataset", token=HF_TOKEN)
    Xtest_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtest.csv", repo_type="dataset", token=HF_TOKEN)
    ytrain_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytrain.csv", repo_type="dataset", token=HF_TOKEN)
    ytest_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytest.csv", repo_type="dataset", token=HF_TOKEN)

    Xtrain = pd.read_csv(Xtrain_local_path)
    Xtest = pd.read_csv(Xtest_local_path)
    ytrain = pd.read_csv(ytrain_local_path).squeeze()
    ytest = pd.read_csv(ytest_local_path).squeeze()
    print("Datasets loaded successfully.")

    numeric_cols = [
        'Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age'
    ]
    categorical_cols = [
        'Product_Sugar_Content', 'Product_Type', 'Store_Id',
        'Store_Size', 'Store_Location_City_Type', 'Store_Type'
    ]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        remainder='drop'
    )

    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    param_grid = {
        'xgbregressor__n_estimators': [100, 150],
        'xgbregressor__max_depth': [3, 4, 5],
        'xgbregressor__learning_rate': [0.05, 0.1],
        'xgbregressor__colsample_bytree': [0.5, 0.7],
        'xgbregressor__reg_lambda': [0.5, 1.0],
    }

    with mlflow.start_run(run_name="Optimized_XGB_Regressor_PROD_Standalone"):
        grid_search = GridSearchCV(
            model_pipeline, param_grid, cv=5, n_jobs=-1,
            scoring='neg_root_mean_squared_error'
        )
        grid_search.fit(Xtrain, ytrain)

        results = grid_search.cv_results_
        for i in range(len(results['params'])):
            param_set = results['params'][i]
            mean_score = results['mean_test_score'][i]
            std_score = results['std_test_score'][i]
            with mlflow.start_run(nested=True):
                mlflow.log_params(param_set)
                mlflow.log_metric("mean_test_neg_rmse", mean_score)
                mlflow.log_metric("std_test_neg_rmse", std_score)

        mlflow.log_params(grid_search.best_params_)
        best_model = grid_search.best_estimator_

        y_pred_train = best_model.predict(Xtrain)
        y_pred_test = best_model.predict(Xtest)

        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(ytrain, y_pred_train))),
            "train_mae": float(mean_absolute_error(ytrain, y_pred_train)),
            "train_r2": float(r2_score(ytrain, y_pred_train)),
            "test_rmse": float(np.sqrt(mean_squared_error(ytest, y_pred_test))),
            "test_mae": float(mean_absolute_error(ytest, y_pred_test)),
            "test_r2": float(r2_score(ytest, y_pred_test)),
        }
        mlflow.log_metrics(metrics)

        print("Logged Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        os.makedirs("mlops/model_building", exist_ok=True)
        model_path = "mlops/model_building/productionmodel.joblib"
        joblib.dump(best_model, model_path)

        mlflow.log_artifact(model_path, artifact_path="model")
        print(f"Model saved as artifact at: {model_path}")
        print("Model Training & Tracking Complete.")
        print(f"Metrics are: {metrics}")

        REPO_TYPE = "model"
        try:
            api.repo_info(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE)
            print(f"Repo '{MODEL_REPO_ID}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Repo '{MODEL_REPO_ID}' not found. Creating new repo...")
            create_repo(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE, private=False)
            print(f"Repo '{MODEL_REPO_ID}' created.")

        print("Uploading to Hugging Face Model Hub..")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="productionmodel.joblib",
            repo_id=MODEL_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print("Optimized model is uploaded to Hugging Face.")

if __name__ == "__main__":
    train_with_grid_search()
