import whisper

from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from whisper.streaming import StreamingConfig
from whisper.tokenizer import get_tokenizer

import torch

from datasets import load_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a single example from the streaming dataset
dataset = load_dataset("MLCommons/peoples_speech", "clean", split="test", streaming=True, trust_remote_code=True)

# You can apply filtering logic here if needed
dataset = dataset.shuffle(seed=42).filter(lambda example: len(example["text"]) > 230)
example = next(iter(dataset))

audio_array = torch.tensor(example["audio"]["array"], device=device)
text_reference = example["text"]

whisper_tiny = whisper.load_model("tiny.en").to(device)

model = whisper.WhisperStreaming(whisper_tiny.dims, StreamingConfig()).to(device)
model.load_state_dict(whisper_tiny.state_dict(), strict=False)

# Load your fine-tuned parameters
model.load_state_dict(torch.load("output/whisper_streaming_pchoose_150.pt"), strict=False)

# Simulate a streaming audio source
def chunk_audio(audio_tensor, chunk_size):
    for i in range(0, len(audio_tensor), chunk_size):
        yield audio_tensor[i:i+chunk_size]

audio = whisper.pad_or_trim(audio_array)

chunk_size = 16000  # 1 second chunks at 16kHz
stream = chunk_audio(audio, chunk_size)

tokenizer = get_tokenizer(multilingual=False)

decoder_input = list(tokenizer.sot_sequence_including_notimestamps)
audio_accum = torch.tensor([], device=device)
stall_count = 0
max_stall_chunks = 3
all_emitted = []

for j, audio_chunk in enumerate(stream, start=1):
    print(f"\n--- Chunk {j} ---")

    audio_chunk = torch.tensor(audio_chunk, device=device)
    audio_accum = torch.cat([audio_accum, audio_chunk])

    padded_audio = pad_or_trim(audio_accum)
    mel_input = log_mel_spectrogram(padded_audio.to(torch.float32)).unsqueeze(0)

    with torch.no_grad():
        h_j = model.encoder(mel_input)

    emitted = False
    while True:
        with torch.no_grad():
            y_in = torch.tensor([decoder_input], device=device)
            logits, p_choose = model.decoder(y_in, h_j)

            if p_choose.ndim == 3:
                _, tgt_len, src_len = p_choose.size()
                p_choose = p_choose.view(model.dims.n_text_layer, -1, tgt_len, src_len)

            p = p_choose[:, :, -1, :].mean().item()

        print(f"Policy probability p = {p:.3f}")

        if p < 0.7 and stall_count < max_stall_chunks:
            print("⚠️ Stalling...")
            stall_count += 1
            break  # wait for more audio in next chunk
        elif p < 0.7:
            next_token = logits[:, -1].argmax(dim=-1).item()
            decoder_input.append(next_token)
        
            if next_token not in {tokenizer.eot, tokenizer.sot}:
                all_emitted.append(next_token)

            print("Decoded so far:", tokenizer.decode(decoder_input))
            break

        emitted = True
        next_token = logits[:, -1].argmax(dim=-1).item()
        decoder_input.append(next_token)
        
        if next_token not in {tokenizer.eot, tokenizer.sot}:
            all_emitted.append(next_token)

        print("Decoded so far:", tokenizer.decode(decoder_input))

        if next_token == tokenizer.eot:
            print("🛑 End of segment")
            decoder_input = list(tokenizer.sot_sequence_including_notimestamps)
            audio_accum = torch.tensor([], device=device)
            stall_count = 0
            break

    if emitted:
        stall_count = 0

# After stream ends, flush remaining audio if any
if len(audio_accum) > 0:
    print("\n🚨 Final flush after last chunk")

    padded_audio = pad_or_trim(audio_accum)
    mel_input = log_mel_spectrogram(padded_audio.to(torch.float32)).unsqueeze(0)

    with torch.no_grad():
        h_j = model.encoder(mel_input)

    while True:
        with torch.no_grad():
            y_in = torch.tensor([decoder_input], device=device)
            logits, _ = model.decoder(y_in, h_j)

        next_token = logits[:, -1].argmax(dim=-1).item()
        decoder_input.append(next_token)

        if next_token not in {tokenizer.eot, tokenizer.sot, tokenizer.no_timestamps}:
            all_emitted.append(next_token)

        print("Decoded so far:", tokenizer.decode(decoder_input))

        if next_token == tokenizer.eot:
            print("🛑 Final end of segment")
            break

# Final output
final_text = tokenizer.decode(all_emitted).strip()
print("\n✅ Final Transcription:")
print(final_text)
print(f"📝 Reference: {text_reference}")
