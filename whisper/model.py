import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import AvgPool1d, ReLU
from torch import Tensor, nn
from typing import Optional, final


from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .streaming import StreamingConfig, transcribe_stream as transcribe_stream_function
from .transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        training: bool = False,
        alpha: Optional[Tensor] = None,
        monotonic_energy: Optional[Tensor] = None,
        is_cross_attn: bool = False,
        beta_weight: Optional[float] = 0.0,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]
            
        wv, qk = self.qkv_attention(q, k, v, mask)

        if training and alpha is not None and monotonic_energy is not None and is_cross_attn:
            cumprod_energy = torch.cumprod(monotonic_energy, dim=-1) + 1e-8 # to prevent div by zero
            inv_cumprod_energy = 1.0 / cumprod_energy
            
            inner = alpha * inv_cumprod_energy
            
            flipped = torch.flip(inner, dims=[-1])
            
            cumsum = torch.cumsum(flipped, dim=-1)
            
            flipped_cumsum = torch.flip(cumsum, dims=[-1])
            
            beta = monotonic_energy * flipped_cumsum
            
            # now we apply beta as soft-attention weights over V
            bsz, T_k, model_dim = v.shape
            D_head =  model_dim // self.n_head
            
            v = v.view(bsz, T_k, self.n_head, D_head).permute(0, 2, 1, 3)
            
            beta_attention = torch.matmul(beta, v).permute(0, 2, 1, 3).flatten(start_dim=2)
            
            wv = (1 - beta_weight) * wv + beta_attention * beta_weight

        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x



class EnergyProjection(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                f"Invalid `num_layers`: {num_layers} for EnergyProjectionLayer."
            )

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                Linear(model_dim, model_dim, bias)
            )
            self.layers.append(ReLU())

    def forward(self, seqs: Tensor) -> Tensor:
        for layer in self.layers:
            seqs = layer(seqs)
        return seqs


@final
class PChooseLayer(nn.Module):
    """Represents a PChoose layer."""

    model_dim: int
    num_heads: int
    energy_bias: nn.Parameter
    monotonic_temperature: float
    q_energy_proj: EnergyProjection
    k_energy_proj: EnergyProjection
    keys_pooling: AvgPool1d

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        streaming_config: StreamingConfig,
        *,
        bias: bool = True,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        :param bias:
            If ``True``, query, key energy projection layers learn an
            additive bias.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        if streaming_config.energy_bias_value != 0.0:
            self.energy_bias = nn.Parameter(
                torch.full([1], streaming_config.energy_bias_value)
            )
        else:
            self.register_module("energy_bias", None)

        self.monotonic_temperature = streaming_config.monotonic_temperature

        if streaming_config.num_monotonic_energy_layers <= 0:
            raise ValueError("Number of monotonic energy layers must be > 0.")

        self.q_energy_proj = EnergyProjection(
            self.model_dim,
            streaming_config.num_monotonic_energy_layers,
            bias,
        )
        self.k_energy_proj = EnergyProjection(
            self.model_dim,
            streaming_config.num_monotonic_energy_layers,
            bias,
        )

        self.keys_pooling = AvgPool1d(
            kernel_size=streaming_config.pre_decision_ratio,
            stride=streaming_config.pre_decision_ratio,
            ceil_mode=True,
        )

    def _monotonic_alignment(self, p):
        prior_size = p.size()
        p = p.flatten(0, 1) # b, t, s
        bsz, tgt_len, src_len = p.size()

        p_ext = p.roll(1, [-1]).unsqueeze(-2).expand(-1, -1, src_len, -1).triu(1) # B, T, S, S

        T = (1 - p_ext).cumprod(-1).triu() # B, T, S, S

        alpha = [p[:, 0] * T[:, 0, 0]]  # First time step: [B, S]

        for i in range(1, tgt_len):
            alpha.append(p[:, i, :] * torch.bmm(alpha[i - 1].unsqueeze(1), T[:, i]).squeeze(1))

        # convert back to b, num_head, t, s
        return torch.reshape(torch.stack(alpha, dim=1), prior_size)

    def forward(self, seqs: Tensor, keys: Tensor) -> Tensor:
        q = self.q_energy_proj(seqs)

        # (N, S, M) -> (N, H, S, K)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        k = self.k_energy_proj(keys)

        # (N, S_p, M) -> (N, H, S_p, K)
        k = k.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        # (N, H, S, K) @ (N, H, K, S_p) = (N, H, S, S_p)
        monotonic_energy = torch.matmul(q, k.transpose(-1, -2))

        monotonic_energy = monotonic_energy * (q.size(-1) ** -0.5)

        if self.energy_bias is not None:
            monotonic_energy += self.energy_bias.to(seqs.dtype)
            
        self.monotonic_energy = monotonic_energy # save for later use, N, H, S, S_p

        # p_choose: (N, H, S, S_p)
        p_choose = torch.sigmoid(monotonic_energy / self.monotonic_temperature)

        return p_choose, self._monotonic_alignment(p_choose)

class MonotonicResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        streaming_config: StreamingConfig,
    ):
        super().__init__()

        self.beta_weight = 0.0
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        
        # p_choose layer shall 
        self.p_choose_layer = PChooseLayer(
            n_state,
            n_head,
            streaming_config,
        )
        
        self.cross_attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
    
    def _cross_attn_forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        training: bool = False,
        ):
        x_norm = self.cross_attn_ln(x)
        p_choose, alpha = self.p_choose_layer(x_norm, xa) # parallel to MonotonicTransformerDecoderLayer class
        x = x + self.cross_attn(x_norm, xa, kv_cache=kv_cache, training=training, alpha=alpha, monotonic_energy=self.p_choose_layer.monotonic_energy, is_cross_attn=True, beta_weight=self.beta_weight)[0]
        
        return x, p_choose, alpha

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        training: bool = False,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]

        x, p_choose, alpha = self._cross_attn_forward(x, xa, mask, kv_cache, training)
        self.alpha = alpha

        x = x + self.mlp(self.mlp_ln(x))

        return x, p_choose


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class MonotonicTextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,
        streaming_config: StreamingConfig,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[MonotonicResidualAttentionBlock] = nn.ModuleList(
            [
                MonotonicResidualAttentionBlock(n_state, n_head, streaming_config)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def decode_with_pchoose(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, training: bool = False):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        p_choose_list: List[Tensor] = []

        for block in self.blocks:
            x, p_choose = block(x, xa, mask=self.mask, kv_cache=kv_cache, training=training)
            p_choose_list.append(p_choose)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        p_choose = torch.cat(p_choose_list, dim=0)
        p_choose = p_choose.flatten(0, 1)

        return logits, p_choose

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, training: bool = False):
        return self.decode_with_pchoose(x, xa, kv_cache, training)


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

class WhisperStreaming(Whisper):
    def __init__(self, dims: ModelDimensions, streaming_config: StreamingConfig):
        super().__init__(dims)
        self.streaming_config = streaming_config
        self.decoder = MonotonicTextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            streaming_config,
        )

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
    transcribe_stream = transcribe_stream_function
