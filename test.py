from datasets import load_dataset
import json
import whisper
from tqdm import tqdm
import os

from whisper.audio import load_audio
from whisper.streaming import StreamingConfig, run_streaming_inference
import logging

import torch

THRESHOLD_PROBABILTY = 0.15

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_tiny = whisper.load_model("tiny.en").to(device)

model = whisper.WhisperStreaming(whisper_tiny.dims, StreamingConfig()).to(device)
model.load_state_dict(whisper_tiny.state_dict(), strict=False)

default = os.path.join(os.path.expanduser("~"), ".cache")
download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")
model.load_state_dict(torch.load(os.path.join(download_root, "whisper_streaming_pchoose_900.pt")), strict=False)


# Load a single example from the streaming dataset
dataset = load_dataset("MLCommons/peoples_speech", "clean", split="test", streaming=True, trust_remote_code=True)

# You can apply filtering logic here if needed
dataset = dataset.filter(lambda example: example["duration_ms"] > 5000)


# Warmup
print("Warming up...")
audio_array = torch.tensor(load_audio('./jfk.wav'), device=device)
run_streaming_inference(model, audio_array, device, threshold_probability=THRESHOLD_PROBABILTY)
print("Done")

batch = []

NUM_EXAMPLES = 200

for i, example in enumerate(tqdm(dataset, total=NUM_EXAMPLES)):
    if i >= NUM_EXAMPLES:
        break
    audio_array = torch.tensor(example["audio"]["array"], device=device)
    text_reference = example["text"]

    try:
        o = run_streaming_inference(model, audio_array, device, threshold_probability=THRESHOLD_PROBABILTY)
    except Exception as e:
        print(i, example["duration_ms"], e)
        o = []

    result = {
        "text": "".join(o).strip(),
        "delays":  [[word, float(j + 1)] for j, s in enumerate(o) for word in s.split()]
    }
    batch.append(result)

    if (i+1) % 25 == 0:
        with open(f"./results/our-900-{THRESHOLD_PROBABILTY}-batch-{i+1}.json", "w") as f:
            json.dump(batch, f)
        batch.clear()
