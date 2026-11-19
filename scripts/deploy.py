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
    fail(f"Could not create {space_id}: {type(exc).__name__}: {getattr(getattr(exc, "response", None), "text", "")[:500]} | {str(exc)[-400:]}")

# Enforce the free "CPU basic" tier. A Space created in the web UI may default to other
# hardware (e.g. ZeroGPU), which refuses to start CPU-only apps, and free accounts are
# not allowed to downgrade it afterwards.
hw = api.get_space_runtime(space_id).requested_hardware
if hw != "cpu-basic":
    print(f"Space hardware is {hw}; requesting cpu-basic (free tier)")
    try:
        api.request_space_hardware(space_id, "cpu-basic")
    except Exception:
        fail(f"{space_id} runs on {hw} and the Hub will not downgrade it to cpu-basic on a free "
             "account. Delete the Space (Settings > Delete this Space) and re-run this workflow; "
             "it will be recreated on cpu-basic.")

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
