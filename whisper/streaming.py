import numpy as np
import torch

from abc import ABC, abstractmethod
from torch import Tensor
from typing import TYPE_CHECKING, Any, Iterator, Optional, List, Tuple, Union
from dataclasses import replace, dataclass
import torch.nn.functional as F

from .decoding import DecodingOptions, DecodingTask, MonotonicPyTorchInference
from .audio import CHUNK_LENGTH, FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, load_audio, log_mel_spectrogram, pad_or_trim

if TYPE_CHECKING:
    from .model import WhisperStreaming

DEBUG = False

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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


class StreamingAgent(ABC):
    @abstractmethod
    def process_iter(self, input: Any, source_finished: bool) -> Optional[Any]:
        pass

    def process(self, inputs: Iterator) -> Iterator:
        try:
            val = next(inputs)
            while val is not Ellipsis:
                try:
                    next_val = next(inputs)
                except StopIteration:
                    next_val = Ellipsis

                o = self.process_iter(val, next_val is Ellipsis)
                if o is not None:
                    yield o

                val = next_val
        except StopIteration:
            debug('STOP ITER', self)


class StreamingAudioFeatureExtractor(StreamingAgent):
    """
    Streaming audio feature extractor that processes audio chunks and extracts Mel spectrograms.
    """
    def __init__(self, n_mels: int = 80, chunk_size: int = HOP_LENGTH, device: Optional[Union[str, torch.device]] = None):
        self.n_mels = n_mels
        self.device = device
        self.chunk_size = chunk_size
        self.previous_residual_samples = np.array([], dtype=np.float32)

    def process_iter(self, input: np.ndarray, source_finished: bool) -> Optional[Tensor]:
        """
        Process streaming audio and yield log-Mel spectrogram chunks.
        Returns (n_mels, n_frames)
        """
        samples = np.concatenate((self.previous_residual_samples, input))
        if len(samples) < self.chunk_size and not source_finished:
            self.previous_residual_samples = samples
            return
        
        input_samples = samples[:self.chunk_size]
        self.previous_residual_samples = samples[self.chunk_size:]

        log_mel_spec = log_mel_spectrogram(input_samples, self.n_mels, device=self.device)

        debug("log mel yield")
        return log_mel_spec

class OfflineAudioEncoder(StreamingAgent):
    def __init__(self, model: "WhisperStreaming", min_length: int = 1, max_length: int = CHUNK_LENGTH-1) -> None:
        self.model = model
        self.mels: Optional[Tensor] = None
        self.min_frames = min_length*FRAMES_PER_SECOND
        self.max_frames = max_length*FRAMES_PER_SECOND

    def process_iter(self, input: Tensor, source_finished: bool) -> Optional[Tensor]:
        """
        Process streaming mels and yields encoder output
        Returns (VEC_SIZE, n_frames)
        """
        if self.mels is None:
            self.mels = input
        else:
            self.mels = torch.cat((self.mels, input), dim=1)
            self.mels = self.mels[:, -self.max_frames:]

        if self.mels.shape[1] < self.min_frames and not source_finished:
            return

        input_mels = pad_or_trim(self.mels.unsqueeze(0), length=N_FRAMES)
        # input_mels = F.pad(self.mels.unsqueeze(0), (0, 3000 - self.mels.shape[-1]))

        debug("encoder len", self.mels.shape)

        return self.model.embed_audio(input_mels)

class StreamingDecoder(StreamingAgent):
    def __init__(self, model: "WhisperStreaming", streaming_config: StreamingConfig,
                 decoding_options: "DecodingOptions", device: Optional[Union[str, torch.device]] = None) -> None:
        self.model = model
        self.streaming_config = streaming_config
        self.task = DecodingTask(model, decoding_options, inference_cls=MonotonicPyTorchInference)
        self.target_sequence: List[int] = []
        self.encoder_output: Tensor = Tensor()
        self.device = device

    def max_len(self, src):
        return self.streaming_config.max_len_a*src.shape[1] + self.streaming_config.max_len_b

    def run_decoder(
        self, tokens: Tensor,
    ) -> Tuple[Tensor, bool, float, Tensor]:
        debug(self.task.tokenizer.decode(tokens.tolist()))
        tokens = tokens.unsqueeze(0)

        logits, p_choose = self.task.inference.decode_with_pchoose(tokens, self.encoder_output)

        # now we need to consider the logits at the last token only
        logits = logits[:, -1]

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in self.task.logit_filters:
            logit_filter.apply(logits, tokens)

        # expand the tokens tensor with the selected next tokens
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=tokens.device)
        updated_tokens, reached_eos = self.task.decoder.update(tokens, logits, sum_logprobs)

        _, tgt_len, src_len = p_choose.size()

        p_choose = p_choose.view(self.model.dims.n_text_layer, -1, tgt_len, src_len)

        if self.streaming_config.decision_method == "min":
            prob = p_choose[self.streaming_config.p_choose_start_layer :, :, -1, -1].min().item()
        elif self.streaming_config.decision_method == "mean":
            prob = p_choose[self.streaming_config.p_choose_start_layer :, :, -1, -1].mean().item()
        else:
            prob = p_choose[self.streaming_config.p_choose_start_layer :, :, -1, -1].median().item()

        return updated_tokens[0], reached_eos, prob, logits

    def postprocess(
        self,
        pred_indices: List[int],
    ) -> str:
        return self.task.tokenizer.decode(pred_indices)

    def process_iter(self, input: Tensor, source_finished: bool) -> Optional[str]:
        debug('READ')
        self.encoder_output = input
        initial_tokens_len = len(self.task.initial_tokens)
        pred_tokens: Tensor = torch.tensor(list(self.task.initial_tokens) + self.target_sequence, device=self.device)
        finished = False
        decoder_features_out = None

        self.task.decoder.reset()
        self.task.inference.cleanup_caching()

        while True:
            updated_tokens, reached_eos, prob, decoder_features = self.run_decoder(pred_tokens)

            if decoder_features_out is None:
                decoder_features_out = decoder_features.new(0)
            decoder_features_out = torch.cat(
                [decoder_features_out, decoder_features], dim=1
            )

            if (
                self.streaming_config.no_early_stop
                and not source_finished
                and (prob < self.streaming_config.decision_threshold or reached_eos)
            ):
                if prob == 1.0:
                    pred_tokens = torch.tensor(self.task.initial_tokens)
                debug('break early stop?', self.streaming_config.no_early_stop, source_finished, prob, reached_eos)
                break

            if (
                finished or reached_eos
                or len(self.target_sequence) + pred_tokens.shape[0] > self.max_len(self.encoder_output)
            ):
                debug('break finished', finished, reached_eos)
                finished = True
                break

            if prob < self.streaming_config.decision_threshold and not source_finished:
                debug('break read', prob, source_finished)
                break

            if (
                len(self.target_sequence) + pred_tokens.shape[0] >= self.max_len(self.encoder_output)
                or pred_tokens.shape[0] >= self.streaming_config.max_consecutive_writes
            ):
                debug('break max_consecutive_writes', prob, source_finished)
                break

            pred_tokens = updated_tokens

        if (pred_tokens.shape[0] - initial_tokens_len) == 0 and not finished:
            return

        newly_pred_tokens = pred_tokens.tolist()[initial_tokens_len:]
        self.target_sequence += newly_pred_tokens

        r = self.postprocess(newly_pred_tokens)
        debug('WRITE', r)
        return r

class SpeechToTextPipeline:
    def __init__(self, model, streaming_config: StreamingConfig, decoding_options: "DecodingOptions",
                 n_mels=80, chunk_size=HOP_LENGTH,
                 min_length=1, max_length=CHUNK_LENGTH-1,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        self.device = device
        self.pipeline = [
            StreamingAudioFeatureExtractor(n_mels=n_mels, chunk_size=chunk_size, device=device),
            OfflineAudioEncoder(model, min_length=min_length, max_length=max_length),
            StreamingDecoder(model, streaming_config=streaming_config,
                             decoding_options=decoding_options, device=device)
        ]

    def process(self, inputs: Iterator) -> Iterator:
        for part in self.pipeline:
            inputs = part.process(inputs)
        return inputs

class AudioStreamSource:
    def __init__(self, audio: Union[np.ndarray, str], chunk_size: int) -> None:
        if isinstance(audio, str):
            audio = load_audio(audio).astype(np.float32)

        self.audio = audio
        self.chunk_size = chunk_size

    def generate(self):
        for i in range(0, len(self.audio), self.chunk_size):
            debug("stream source yield")
            yield self.audio[i:i+self.chunk_size]


def transcribe_stream(
    model: "WhisperStreaming",
    audio: Union[str, np.ndarray],
    decoding_options: DecodingOptions = DecodingOptions(),
    streaming_options: StreamingConfig = StreamingConfig(),
    min_length=1, max_length=CHUNK_LENGTH-1, chunk_size=10*HOP_LENGTH,
    **kwargs,
) -> str:
    if kwargs:
        decoding_options = replace(decoding_options, **kwargs)
        streaming_options = replace(streaming_options, **kwargs)

    source = AudioStreamSource(audio, chunk_size)

    pipeline = SpeechToTextPipeline(model, streaming_options, decoding_options,
                                    n_mels=model.dims.n_mels,
                                    chunk_size=chunk_size, min_length=min_length,
                                    max_length=max_length, device=model.device)
    outputs = pipeline.process(source.generate())

    return "".join(outputs)
