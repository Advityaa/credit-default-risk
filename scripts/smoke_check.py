"""Wait for the Space to finish building, then call its /predict API."""
import os
import sys
import time

from gradio_client import Client
from huggingface_hub import HfApi

space_id = os.environ["HF_SPACE_ID"]
api = HfApi(token=os.environ.get("HF_TOKEN"))
deadline = time.time() + 15 * 60
time.sleep(30)  # give the Hub time to register the new commit and start a build
while True:
    stage = api.get_space_runtime(space_id).stage
    print("Space stage:", stage, flush=True)
    if stage == "RUNNING":
        break
    if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"} or time.time() > deadline:
        sys.exit(f"Space failed to start: {stage}")
    time.sleep(20)

client = Client(space_id, token=os.environ.get("HF_TOKEN"), verbose=False)
probs, text = client.predict(
    50000, "Female", "University", "Single", 35,
    *["2 months late (2)"] * 6, *[48000] * 6, *[0] * 6, api_name="/predict")
print(text)
assert "HIGH RISK" in text, "smoke test: chronically late client was not flagged"
print("Smoke test passed")
