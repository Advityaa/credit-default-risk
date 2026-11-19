"""Upload the app and freshly trained model to the Hugging Face Space.

The Hub's HTTP API is used instead of `git push` so the binary model artifact
does not have to be committed to the GitHub repository or tracked with LFS.
Failures are reported as GitHub Actions `::error::` annotations so the cause is
visible on the run summary page without opening the raw logs.
"""
import os
import sys

from huggingface_hub import HfApi


def fail(msg: str) -> None:
    print(f"::error title=Deploy failed::{msg}")
    sys.exit(1)


space_id = os.environ.get("HF_SPACE_ID", "").strip()
token = os.environ.get("HF_TOKEN", "").strip()
if not space_id:
    fail("Repository variable HF_SPACE_ID is not set (Settings > Secrets and variables > Actions > Variables).")
if not token:
    fail("Repository secret HF_TOKEN is empty or missing (Settings > Secrets and variables > Actions > Secrets).")

api = HfApi(token=token)
try:
    who = api.whoami()
except Exception as exc:  # invalid or revoked token
    fail(f"HF_TOKEN was rejected by the Hub ({type(exc).__name__}). Create a new token with Write access.")
role = who.get("auth", {}).get("accessToken", {}).get("role", "unknown")
print(f"Authenticated as {who.get('name')} (token role: {role})")
if role == "read":
    fail(f"HF_TOKEN for user {who.get('name')} is read-only; create a token with Write access.")

sha = os.environ.get("GITHUB_SHA", "local")[:7]
try:
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
except Exception as exc:
    body = getattr(getattr(exc, "response", None), "text", "")[:500]
    fail(f"Could not create {space_id}: {type(exc).__name__}: {body}")

# Cost guard: only deploy to hardware that is free for this account. CPU basic is free;
# ZeroGPU is free within the account's daily GPU quota and needs an @spaces.GPU function
# (present in app.py). Anything else would be billed, so refuse to deploy.
FREE_HARDWARE = {"cpu-basic", "zero-a10g"}
hw = api.get_space_runtime(space_id).requested_hardware
print(f"Space hardware: {hw}")
if hw not in FREE_HARDWARE:
    fail(f"{space_id} is on paid hardware ({hw}); switch it to a free tier before deploying.")

try:
    info = api.upload_folder(
        repo_id=space_id, repo_type="space", folder_path=".",
        allow_patterns=["app.py", "features.py", "requirements.txt", "README.md", "model/*"],
        delete_patterns=["model/*"],
        commit_message=f"Deploy {sha} from GitHub Actions",
    )
except Exception as exc:
    detail = str(exc).replace("\n", " ")[:400]
    fail(f"Upload to {space_id} as {who.get('name')} (role {role}) failed: {type(exc).__name__}: {detail}")
print(f"Deployed commit {info.oid} to https://huggingface.co/spaces/{space_id}")
