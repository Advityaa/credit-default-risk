"""Upload the app and freshly trained model to the Hugging Face Space.

The Hub's HTTP API is used instead of `git push` so the binary model artifact
does not have to be committed to the GitHub repository or tracked with LFS.
"""
import os
import sys

from huggingface_hub import HfApi

space_id = os.environ["HF_SPACE_ID"]
sha = os.environ.get("GITHUB_SHA", "local")[:7]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
info = api.upload_folder(
    repo_id=space_id, repo_type="space", folder_path=".",
    allow_patterns=["app.py", "features.py", "requirements.txt", "README.md", "model/*"],
    delete_patterns=["model/*"],
    commit_message=f"Deploy {sha} from GitHub Actions",
)
print(f"Deployed commit {info.oid} to https://huggingface.co/spaces/{space_id}", file=sys.stderr)
