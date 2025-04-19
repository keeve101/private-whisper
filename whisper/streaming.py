import torch
import logging

from whisper.tokenizer import get_tokenizer

logger = logging.getLogger("__main__")

from dataclasses import dataclass

from .audio import log_mel_spectrogram, pad_or_trim

@dataclass
class StreamingConfig:
    monotonic_temperature: float = 0.2
    num_monotonic_energy_layers: int = 4
    pre_decision_ratio: int = 2
    energy_bias_value: float = -0.5

    p_choose_start_layer: int = 0
    decision_method = "min"
    decision_threshold: float = 0.5

    max_len_a = 1
    max_len_b = 200
    max_consecutive_writes = 50
    no_early_stop = False

# Simulate a streaming audio source
def chunk_audio(audio_tensor, chunk_size):
    for i in range(0, len(audio_tensor), chunk_size):
        yield audio_tensor[i:i+chunk_size]


def run_streaming_inference(model, audio, device='auto', threshold_probability = 0.17):
    
    chunk_size = 16000  # 1 second chunks at 16kHz
    stream = chunk_audio(audio, chunk_size)

    tokenizer = get_tokenizer(multilingual=False)

    decoder_input = list(tokenizer.sot_sequence_including_notimestamps)
    audio_accum = torch.tensor([], device=device)
    stall_count = 0
    max_stall_chunks = 3
    all_emitted = []
    emitted_chunks = []

    for j, audio_chunk in enumerate(stream, start=1):
        logger.debug(f"--- Chunk {j} ---")

        emitted_len = len(all_emitted)

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

            logger.debug(f"Policy probability p = {p:.3f}")

            if p < threshold_probability and stall_count < max_stall_chunks:
                logger.debug("⚠️ Stalling...")
                stall_count += 1
                break  # wait for more audio in next chunk
            elif p < threshold_probability:
                next_token = logits[:, -1].argmax(dim=-1).item()
                decoder_input.append(next_token)
            
                if next_token not in {tokenizer.eot, tokenizer.sot}:
                    all_emitted.append(next_token)

                logger.debug("Decoded so far:", tokenizer.decode(decoder_input))
                break

            emitted = True
            next_token = logits[:, -1].argmax(dim=-1).item()
            decoder_input.append(next_token)
            
            if next_token not in {tokenizer.eot, tokenizer.sot}:
                all_emitted.append(next_token)

            logger.debug("Decoded so far:", tokenizer.decode(decoder_input))

            if next_token == tokenizer.eot:
                logger.debug("🛑 End of segment")
                decoder_input = list(tokenizer.sot_sequence_including_notimestamps)
                audio_accum = torch.tensor([], device=device)
                stall_count = 0
                break

        emitted_chunks.append(tokenizer.decode(all_emitted[emitted_len:]))
        if emitted:
            stall_count = 0

    # After stream ends, flush remaining audio if any
    if len(audio_accum) > 0:
        emitted_len = len(all_emitted)
        logger.debug("🚨 Final flush after last chunk")

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

            logger.debug("Decoded so far:", tokenizer.decode(decoder_input))

            if next_token == tokenizer.eot:
                logger.debug("🛑 Final end of segment")
                break

        emitted_chunks.append(tokenizer.decode(all_emitted[emitted_len:]))

    # Final output
    return emitted_chunks
