from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Hugging Face Repository details
REPO_ID = "flyingdragon98/salesforecast"
REPO_TYPE = "dataset"
DATA_FOLDER = "mlops/data"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

def register_data():
    # Step 1: Check if the repository exists, if not, create it
    try:
        api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"Repository '{REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repository '{REPO_ID}' not found. Creating new repository...")
        create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
        print(f"Repository '{REPO_ID}' created.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Step 2: Upload the folder (contains SuperKart.csv)
    print(f"Uploading data from {DATA_FOLDER} to Hugging Face...")
    try:
        api.upload_folder(
            folder_path=DATA_FOLDER,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"Data successfully registered at: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    register_data()
