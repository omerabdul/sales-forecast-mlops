import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

# CONFIGURATION
REPO_ID = "flyingdragon98/salesforecast"
DATA_DIR = "mlops/data"
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = f"hf://datasets/{REPO_ID}/SuperKart.csv"

def prepare_data():
    print("Step 1: Loading dataset from Hugging Face...")
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded from hugging face successfully.")

    # 2.1 Remove unnecessary columns
    print("Step 2: Performing data cleaning...")
    # Product_Id is a near-unique identifier with no standalone predictive
    # signal, so it is dropped. Store_Id is KEPT: it only takes ~4 distinct
    # values here and captures store-specific baseline performance that
    # Store_Size/Store_Type/Store_Location_City_Type don't fully explain.
    cols_to_drop = ['Product_Id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 2.2 Standardize categorical values
    if 'Product_Sugar_Content' in df.columns:
        sugar_map = {
            'low sugar': 'Low Sugar', 'LS': 'Low Sugar', 'Low Sugar': 'Low Sugar',
            'reg': 'Regular', 'Regular': 'Regular', 'Regular ': 'Regular',
            'no sugar': 'No Sugar', 'NS': 'No Sugar', 'No Sugar': 'No Sugar',
        }
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].astype(str).str.strip().replace(sugar_map)

    # 2.3 Impute missing values
    if 'Product_Weight' in df.columns:
        df['Product_Weight'] = df.groupby('Product_Type')['Product_Weight'] \
            .transform(lambda s: s.fillna(s.median()))

    if 'Store_Size' in df.columns:
        df['Store_Size'] = df.groupby('Store_Type')['Store_Size'] \
            .transform(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else 'Medium'))

    # 2.4 Feature engineering
    if 'Store_Establishment_Year' in df.columns:
        current_year = datetime.now().year
        df['Store_Age'] = current_year - df['Store_Establishment_Year']
        df = df.drop(columns=['Store_Establishment_Year'])

    # NOTE: categorical columns are intentionally NOT label-encoded here.
    # They are left as clean strings and one-hot encoded inside the training
    # pipeline (train.py) via ColumnTransformer, so the same fitted encoder
    # is reused consistently at inference time in app.py.

    print("Step 3: Splitting into train and test sets...")

    target_col = 'Product_Store_Sales_Total'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Regression target -> no stratify (stratify only applies to classification labels)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42)

    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {
        "Xtrain.csv": os.path.join(DATA_DIR, "Xtrain.csv"),
        "Xtest.csv": os.path.join(DATA_DIR, "Xtest.csv"),
        "ytrain.csv": os.path.join(DATA_DIR, "ytrain.csv"),
        "ytest.csv": os.path.join(DATA_DIR, "ytest.csv")
    }

    for name, path in paths.items():
        obj = eval(name.split(".")[0])
        if isinstance(obj, pd.Series):
            obj.to_csv(path, index=False, header=True)
        else:
            obj.to_csv(path, index=False)

    print(f"Local files saved in {DATA_DIR}")

    print("Step 4: Uploading processed data back to Hugging Face...")
    api = HfApi(token=HF_TOKEN)

    for filename, local_path in paths.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    print("Preprocessing complete. Train/Test sets uploaded to Hugging Face.")

if __name__ == "__main__":
    prepare_data()
