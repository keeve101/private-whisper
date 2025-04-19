import whisper
from whisper.streaming import StreamingConfig

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_tiny = whisper.load_model("tiny.en").to(device)

print(whisper_tiny.transcribe('./jfk.wav'))

model = whisper.WhisperStreaming(whisper_tiny.dims, StreamingConfig()).to(device)
model.load_state_dict(whisper_tiny.state_dict(), strict=False)

# Load your fine-tuned parameters
model.load_state_dict(torch.load("whisper_streaming_pchoose.pt"), strict=False)

print(model.transcribe_stream('./jfk.wav', chunk_size=10000))
