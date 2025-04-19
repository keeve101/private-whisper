import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.streaming import StreamingConfig
from whisper.tokenizer import get_tokenizer
from whisper.normalizers import EnglishTextNormalizer

import evaluate

from datasets import load_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load streaming dataset
dataset = load_dataset("MLCommons/peoples_speech", "clean", split="test", streaming=True, trust_remote_code=True)
dataset = dataset.shuffle(seed=42).filter(lambda example: len(example["text"]) > 230)

# Load models and tokenizer
whisper_tiny = whisper.load_model("tiny.en").to(device)
model = whisper.WhisperStreaming(whisper_tiny.dims, StreamingConfig()).to(device)


model_path = r"C:\Users\keithh\Downloads\new\whisper_streaming_pchoose_900.pt"
model.load_state_dict(torch.load(model_path), strict=False)
tokenizer = get_tokenizer(multilingual=False)

def chunk_audio(audio_tensor, chunk_size):
    for i in range(0, len(audio_tensor), chunk_size):
        yield audio_tensor[i:i+chunk_size]

def process_example(example, idx):
    print(f"\n\n=================== EXAMPLE {idx + 1} ===================")

    audio_array = torch.tensor(example["audio"]["array"], device=device)
    text_reference = example["text"]

    audio = pad_or_trim(audio_array)
    chunk_size = 16000
    stream = chunk_audio(audio, chunk_size)

    decoder_input = list(tokenizer.sot_sequence_including_notimestamps)
    audio_accum = torch.tensor([], device=device)
    audio_accum_time = 0.0
    stall_count = 0
    max_stall_chunks = 3
    all_emitted = []
    emission_timestamps = []

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

            if p < 0.17 and stall_count < max_stall_chunks:
                print("⚠️ Stalling...")
                stall_count += 1
                break
            elif p < 0.17:
                next_token = logits[:, -1].argmax(dim=-1).item()
                decoder_input.append(next_token)

                if next_token not in {tokenizer.no_timestamps, tokenizer.sot, tokenizer.eot}:
                    seconds_seen = len(audio_accum) / 16000 + audio_accum_time
                    emission_timestamps.append(seconds_seen)

                all_emitted.append(next_token)
                print("Decoded so far:", tokenizer.decode(decoder_input))
                break

            emitted = True
            next_token = logits[:, -1].argmax(dim=-1).item()
            decoder_input.append(next_token)

            if next_token not in {tokenizer.no_timestamps, tokenizer.sot, tokenizer.eot}:
                seconds_seen = len(audio_accum) / 16000 + audio_accum_time
                emission_timestamps.append(seconds_seen)

            all_emitted.append(next_token)
            print("Decoded so far:", tokenizer.decode(decoder_input))

            if next_token == tokenizer.eot:
                print("🛑 End of segment")
                decoder_input = list(tokenizer.sot_sequence_including_notimestamps)
                audio_accum_time += len(audio_accum) / 16000
                audio_accum = torch.tensor([], device=device)
                stall_count = 0
                break

        if emitted:
            stall_count = 0

    # Final flush
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

            if next_token not in {tokenizer.no_timestamps, tokenizer.sot, tokenizer.eot}:
                seconds_seen = len(audio_accum) / 16000 + audio_accum_time
                emission_timestamps.append(seconds_seen)

            all_emitted.append(next_token)
            print("Decoded so far:", tokenizer.decode(decoder_input))

            if next_token == tokenizer.eot:
                print("🛑 Final end of segment")
                break

    # Compute final transcript and average lag
    filtered_tokens = [
        token for token in all_emitted
        if token not in {tokenizer.no_timestamps, tokenizer.sot, tokenizer.eot}
    ]
    final_text = tokenizer.decode(filtered_tokens).strip()
    all_words = final_text.split()

    word_idx_to_token_idx = {}
    token_idx = 0

    for word_idx, word in enumerate(all_words):
        word_idx_to_token_idx[word_idx] = []

        current_subword = ""
        while token_idx < len(filtered_tokens):
            token = filtered_tokens[token_idx]
            decoded_token = tokenizer.decode([token]).strip()

            if decoded_token == "":
                token_idx += 1
                continue

            current_subword += decoded_token
            word_idx_to_token_idx[word_idx].append(token_idx)
            token_idx += 1

            if current_subword == word:
                break

    word_emission_times = {}

    for word_idx, token_indices in word_idx_to_token_idx.items():
        try:
            timestamps = [emission_timestamps[i] for i in token_indices if i < len(emission_timestamps)]
            if timestamps:
                word_emission_times[word_idx] = min(timestamps)
        except IndexError:
            continue

    result = whisper_tiny.transcribe(audio_array.to(torch.float32), word_timestamps=True)
    whisper_pred = result["text"].strip()
    token_audio_times = []
    
    for segment in result["segments"]:
        for word_info in segment["words"]:
            token_audio_times.extend([word_info["start"]])
            
    word_emission_times = list(word_emission_times.values())
    min_len = min(len(word_emission_times), len(token_audio_times))

    lags = [
        emission_timestamps[i] - token_audio_times[i]
        for i in range(min_len)
    ]

    # Drop upper quartile (keep 25th to 75th percentile)
    lags_sorted = sorted(lags)
    cut_lags = lags_sorted[len(lags)//4 : 3*len(lags)//4]
    average_lagging = sum(cut_lags) / len(cut_lags) if cut_lags else float('nan')
    
    print("\n✅ Final Transcription:")
    print(final_text)

    print("\n✅ Whisper Pred:")
    print(whisper_pred)
    
    print(f"📝 Reference: {text_reference}")
    print(f"📉 AL: {average_lagging:.3f}\n")
    
    return final_text, whisper_pred, text_reference, average_lagging

# ==== MAIN LOOP =====
num_examples = 15

whisper_streaming_preds = []
whisper_preds = []
text_refs = []
average_laggings = []

sacrebleu_metric = evaluate.load("sacrebleu")
wer_metric = evaluate.load("wer")
normalizer = EnglishTextNormalizer()

for i, example in enumerate(dataset):
    if i >= num_examples:
        break
    whisper_streaming_pred, whisper_pred, text_ref, average_lagging = process_example(example, i)
    
    whisper_streaming_preds.append(normalizer(whisper_streaming_pred))
    whisper_preds.append(normalizer(whisper_pred))
    text_refs.append(normalizer(text_ref))
    average_laggings.append(average_lagging)

whisper_streaming_bleu_score = sacrebleu_metric.compute(predictions=whisper_streaming_preds, references=text_refs)
print(f"SacreBLEU WhisperStreaming: {whisper_streaming_bleu_score['score']:.3f}")

whisper_streaming_wer_score = wer_metric.compute(predictions=whisper_streaming_preds, references=text_refs)
print(f"WER WhisperStreaming: {whisper_streaming_wer_score:.3f}")

whisper_tiny_bleu_score = sacrebleu_metric.compute(predictions=whisper_preds, references=text_refs)
print(f"SacreBLEU Whisper: {whisper_tiny_bleu_score['score']:.3f}")

whisper_tiny_wer_score = wer_metric.compute(predictions=whisper_preds, references=text_refs)
print(f"WER Whisper: {whisper_tiny_wer_score:.3f}")

print(f"Average Lagging: {sum(average_laggings) / len(average_laggings):.3f}")
