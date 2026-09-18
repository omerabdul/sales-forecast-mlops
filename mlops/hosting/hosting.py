from huggingface_hub import HfApi
import os

# CONFIGURATION
# This is the Space where the Streamlit UI will live
REPO_ID = "flyingdragon98/superkart"

DEPLOY_FOLDER = "mlops/deployment"

def deploy():
    api = HfApi()
    # 1. Ensure the Space exists (Set to Docker SDK, since we ship a custom Dockerfile)
    print(f"Checking if Space {REPO_ID} exists...")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
    except Exception as e:
        print(f"Note on repo creation: {e}")

    # 2. Upload the entire deployment folder
    print(f"🚀 Uploading contents of {DEPLOY_FOLDER} to Hugging Face Space...")
    api.upload_folder(
        folder_path=DEPLOY_FOLDER,
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo="",
    )

    print(f"✅ Deployment successful! View app at: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    deploy()
