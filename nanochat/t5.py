"""
Minimal T5-style encoder-decoder Transformer.

The goal here is the same as `gpt.py`: keep the implementation compact,
 readable, and immediately useful for experiments.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
import nanochat.flash_attention as flash_attention_module
from nanochat.flash_attention import flash_attn
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class T5Config:
    sequence_len: int = 2048
    encoder_sequence_len: int | None = None
    decoder_sequence_len: int | None = None
    vocab_size: int = 32768
    n_layer: int = 13
    n_encoder_layer: int | None = None
    n_decoder_layer: int | None = None
    n_head: int = 12
    n_kv_head: int | None = None
    n_embd: int = 768
    ff_mult: float = 4.0
    ff_hidden_size: int | None = None
    feed_forward: str = "gated-gelu"
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    layer_norm_epsilon: float = 1e-6

    def __post_init__(self):
        if self.encoder_sequence_len is None:
            self.encoder_sequence_len = self.sequence_len
        if self.decoder_sequence_len is None:
            self.decoder_sequence_len = self.sequence_len
        if self.n_encoder_layer is None:
            self.n_encoder_layer = self.n_layer
        if self.n_decoder_layer is None:
            self.n_decoder_layer = self.n_layer
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.ff_hidden_size is None:
            self.ff_hidden_size = int(self.ff_mult * self.n_embd)


def dropout(x, p, training):
    return F.dropout(x, p=p, training=training) if p > 0 else x


def apply_sequence_mask(x, attention_mask):
    if attention_mask is None:
        return x
    return x * attention_mask.to(dtype=x.dtype).unsqueeze(-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(dtype=x.dtype), self.eps)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match activation dtype in forward."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))

class RelativePositionBias(nn.Module):
    def __init__(self, config, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, config.n_head)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
        buckets = 0
        if bidirectional:
            num_buckets //= 2
            buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = relative_position.abs()
        else:
            relative_position = (-relative_position).clamp_min(0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6)
            / torch.log(torch.tensor(max_distance / max_exact, device=relative_position.device))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = relative_position_if_large.clamp(max=num_buckets - 1)
        buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return buckets

    def forward(self, query_length, key_length, device, query_offset=0):
        context_position = query_offset + torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout = config.dropout

    def _shape(self, x, n_head):
        B, T, _ = x.size()
        return x.view(B, T, n_head, self.head_dim)

    def forward(
        self,
        x,
        key_value_states=None,
        attention_mask=None,
        position_bias=None,
        causal=False,
        past_key_value=None,
        use_cache=False,
    ):
        B, T, _ = x.size()
        source = x if key_value_states is None else key_value_states

        q = self._shape(self.c_q(x), self.n_head)
        if past_key_value is not None and key_value_states is not None:
            k, v = past_key_value
        else:
            k = self._shape(self.c_k(source), self.n_kv_head)
            v = self._shape(self.c_v(source), self.n_kv_head)
            if past_key_value is not None and key_value_states is None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v) if use_cache else None
        key_mask = None
        if attention_mask is not None:
            mask_is_dense = attention_mask.to(torch.bool).all()
            if not mask_is_dense:
                key_mask = attention_mask[:, None, None, :]
        y = flash_attn.flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            attn_mask=key_mask,
            attn_bias=position_bias,
            dropout_p=self.dropout if self.training else 0.0,
            require_selected_backend=flash_attention_module.FAST_ATTN_BACKEND is not None,
        )
        y = y.contiguous().view(B, T, self.n_embd)
        y = self.c_proj(y)
        y = dropout(y, self.dropout, self.training)
        return y, present_key_value


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.ff_hidden_size
        self.wi = Linear(config.n_embd, hidden_size, bias=False)
        self.wi_gate = Linear(config.n_embd, hidden_size, bias=False) if config.feed_forward.startswith("gated-") else None
        self.wo = Linear(hidden_size, config.n_embd, bias=False)
        self.feed_forward = config.feed_forward
        self.dropout = config.dropout

    def _activate(self, x):
        if self.feed_forward.endswith("relu"):
            return F.relu(x)
        if self.feed_forward.endswith("gelu"):
            return F.gelu(x, approximate="tanh")
        if self.feed_forward.endswith("silu"):
            return F.silu(x)
        raise ValueError(f"Unsupported feed_forward: {self.feed_forward}")

    def forward(self, x):
        y = self.wi(x)
        if self.wi_gate is not None:
            y = y * self._activate(self.wi_gate(x))
        else:
            y = self._activate(y)
        y = self.wo(y)
        return dropout(y, self.dropout, self.training)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn_norm = RMSNorm(config.n_embd, config.layer_norm_epsilon)
        self.ff_norm = RMSNorm(config.n_embd, config.layer_norm_epsilon)
        self.self_attn = Attention(config)
        self.ff = FeedForward(config)

    def forward(self, x, attention_mask, position_bias):
        y, _ = self.self_attn(
            self.self_attn_norm(x),
            attention_mask=attention_mask,
            position_bias=position_bias,
            causal=False,
        )
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn_norm = RMSNorm(config.n_embd, config.layer_norm_epsilon)
        self.cross_attn_norm = RMSNorm(config.n_embd, config.layer_norm_epsilon)
        self.ff_norm = RMSNorm(config.n_embd, config.layer_norm_epsilon)
        self.self_attn = Attention(config)
        self.cross_attn = Attention(config)
        self.ff = FeedForward(config)

    def forward(
        self,
        x,
        encoder_hidden_states,
        self_attention_mask,
        encoder_attention_mask,
        self_position_bias,
        past_key_value=None,
        use_cache=False,
    ):
        y, present = self.self_attn(
            self.self_attn_norm(x),
            attention_mask=self_attention_mask,
            position_bias=self_position_bias,
            causal=True,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + y
        y, _ = self.cross_attn(
            self.cross_attn_norm(x),
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            causal=False,
        )
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        return x, present


class T5(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.max_seq_len = config.decoder_sequence_len
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.shared = nn.Embedding(padded_vocab_size, config.n_embd)
        self.encoder = nn.ModuleDict({
            "h": nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layer)]),
            "ln_f": RMSNorm(config.n_embd, config.layer_norm_epsilon),
        })
        self.decoder = nn.ModuleDict({
            "h": nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layer)]),
            "ln_f": RMSNorm(config.n_embd, config.layer_norm_epsilon),
        })
        self.encoder_relative_position_bias = RelativePositionBias(config, bidirectional=True)
        self.decoder_relative_position_bias = RelativePositionBias(config, bidirectional=False)
        self.lm_head = None if config.tie_word_embeddings else Linear(config.n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.shared.weight, mean=0.0, std=0.8)
        if self.lm_head is not None:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.encoder.h:
            for proj in [block.self_attn.c_q, block.self_attn.c_k, block.self_attn.c_v]:
                torch.nn.init.uniform_(proj.weight, -s, s)
            torch.nn.init.zeros_(block.self_attn.c_proj.weight)
            torch.nn.init.uniform_(block.ff.wi.weight, -s * 0.4, s * 0.4)
            if block.ff.wi_gate is not None:
                torch.nn.init.uniform_(block.ff.wi_gate.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.ff.wo.weight)

        for block in self.decoder.h:
            for proj in [block.self_attn.c_q, block.self_attn.c_k, block.self_attn.c_v]:
                torch.nn.init.uniform_(proj.weight, -s, s)
            torch.nn.init.zeros_(block.self_attn.c_proj.weight)
            for proj in [block.cross_attn.c_q, block.cross_attn.c_k, block.cross_attn.c_v]:
                torch.nn.init.uniform_(proj.weight, -s, s)
            torch.nn.init.zeros_(block.cross_attn.c_proj.weight)
            torch.nn.init.uniform_(block.ff.wi.weight, -s * 0.4, s * 0.4)
            if block.ff.wi_gate is not None:
                torch.nn.init.uniform_(block.ff.wi_gate.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.ff.wo.weight)

        torch.nn.init.zeros_(self.encoder_relative_position_bias.relative_attention_bias.weight)
        torch.nn.init.zeros_(self.decoder_relative_position_bias.relative_attention_bias.weight)

        for module in self.modules():
            if isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

        if COMPUTE_DTYPE != torch.float16:
            self.shared.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.shared.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        excluded = self.shared.weight.numel()
        if self.lm_head is not None:
            excluded += self.lm_head.weight.numel()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t_enc = self.config.encoder_sequence_len
        t_dec = self.config.decoder_sequence_len
        attn_flops = (
            self.config.n_encoder_layer * (12 * h * q * t_enc) +
            self.config.n_decoder_layer * (12 * h * q * t_dec + 12 * h * q * t_enc)
        )
        return 6 * (nparams - excluded) + attn_flops

    def num_scaling_params(self):
        shared = self.shared.weight.numel()
        lm_head = 0 if self.lm_head is None else self.lm_head.weight.numel()
        encoder = sum(p.numel() for p in self.encoder.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        relpos = (
            self.encoder_relative_position_bias.relative_attention_bias.weight.numel()
            + self.decoder_relative_position_bias.relative_attention_bias.weight.numel()
        )
        total = shared + lm_head + encoder + decoder + relpos
        assert total == sum(p.numel() for p in self.parameters())
        return {
            "shared": shared,
            "lm_head": lm_head,
            "encoder": encoder,
            "decoder": decoder,
            "relative_attention_bias": relpos,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        special_ids = {id(self.shared.weight)}
        embedding_params = [self.shared.weight]
        lm_head_params = []
        if self.lm_head is not None:
            lm_head_params.append(self.lm_head.weight)
            special_ids.add(id(self.lm_head.weight))

        named_params = list(self.named_parameters())
        matrix_params = [
            p for name, p in named_params
            if p.ndim >= 2 and id(p) not in special_ids and "relative_attention_bias" not in name
        ]
        relative_bias_params = [
            p for name, p in named_params
            if "relative_attention_bias" in name
        ]
        scalar_params = [
            p for name, p in named_params
            if p.ndim < 2 and id(p) not in special_ids
        ]

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=relative_bias_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.99), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=scalar_params, lr=scalar_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="muon",
                params=group_params,
                lr=matrix_lr,
                momentum=0.95,
                ns_steps=5,
                beta2=0.9,
                weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _embed(self, ids):
        x = self.shared(ids)
        return x.to(COMPUTE_DTYPE)

    def _project(self, x):
        if self.lm_head is None:
            logits = F.linear(x, self.shared.weight.to(dtype=x.dtype))
        else:
            logits = self.lm_head(x)
        return logits[..., :self.config.vocab_size].float()

    def _shift_right(self, targets):
        targets = torch.where(targets >= 0, targets, torch.full_like(targets, self.config.pad_token_id))
        decoder_input_ids = torch.empty_like(targets)
        decoder_input_ids[:, 0] = self.config.pad_token_id
        decoder_input_ids[:, 1:] = targets[:, :-1]
        return decoder_input_ids

    def _shift_right_attention_mask(self, targets):
        attention_mask = targets >= 0
        shifted = torch.zeros_like(attention_mask)
        shifted[:, 0] = attention_mask.any(dim=1)
        shifted[:, 1:] = attention_mask[:, :-1]
        return shifted

    def _infer_encoder_attention_mask(self, input_ids, attention_mask):
        if attention_mask is not None:
            return attention_mask.to(dtype=torch.bool)
        attention_mask = input_ids.ne(self.config.pad_token_id)
        if attention_mask.size(1) > 0:
            attention_mask[:, 0] = True
        return attention_mask

    def _infer_decoder_attention_mask(self, decoder_input_ids, decoder_attention_mask):
        if decoder_attention_mask is not None:
            return decoder_attention_mask.to(dtype=torch.bool)
        attention_mask = decoder_input_ids.ne(self.config.pad_token_id)
        if attention_mask.size(1) > 0:
            attention_mask[:, 0] = True
        return attention_mask

    def encode(self, input_ids, attention_mask=None):
        B, T = input_ids.size()
        assert T <= self.config.encoder_sequence_len, f"Encoder sequence too long: {T} > {self.config.encoder_sequence_len}"
        attention_mask = self._infer_encoder_attention_mask(input_ids, attention_mask)
        x = apply_sequence_mask(dropout(self._embed(input_ids), self.config.dropout, self.training), attention_mask)
        position_bias = self.encoder_relative_position_bias(T, T, x.device)
        for block in self.encoder.h:
            x = block(x, attention_mask, position_bias)
            x = apply_sequence_mask(x, attention_mask)
        return apply_sequence_mask(self.encoder.ln_f(x), attention_mask)

    def decode_hidden(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        B, T = decoder_input_ids.size()
        decoder_attention_mask = self._infer_decoder_attention_mask(decoder_input_ids, decoder_attention_mask)
        past_length = 0 if not past_key_values else past_key_values[0][0].size(1)
        total_length = past_length + T
        assert total_length <= self.config.decoder_sequence_len, f"Decoder sequence too long: {total_length} > {self.config.decoder_sequence_len}"
        if past_length > 0:
            past_mask = torch.ones(B, past_length, dtype=torch.bool, device=decoder_attention_mask.device)
            self_attention_mask = torch.cat([past_mask, decoder_attention_mask], dim=1)
        else:
            self_attention_mask = decoder_attention_mask
        x = apply_sequence_mask(dropout(self._embed(decoder_input_ids), self.config.dropout, self.training), decoder_attention_mask)
        self_position_bias = self.decoder_relative_position_bias(T, total_length, x.device, query_offset=past_length)
        presents = []
        for layer_idx, block in enumerate(self.decoder.h):
            past = None if past_key_values is None else past_key_values[layer_idx]
            x, present = block(
                x,
                encoder_hidden_states,
                self_attention_mask,
                encoder_attention_mask,
                self_position_bias,
                past_key_value=past,
                use_cache=use_cache,
            )
            x = apply_sequence_mask(x, decoder_attention_mask)
            if use_cache:
                presents.append(present)
        x = apply_sequence_mask(self.decoder.ln_f(x), decoder_attention_mask)
        return x, presents if use_cache else None

    def decode(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        hidden, presents = self.decode_hidden(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self._project(hidden)
        return logits, presents

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        targets=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        loss_reduction="mean",
    ):
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_attention_mask = attention_mask
        if encoder_hidden_states is None:
            assert input_ids is not None, "input_ids are required unless encoder_hidden_states are provided"
            encoder_attention_mask = self._infer_encoder_attention_mask(input_ids, attention_mask)
            encoder_hidden_states = self.encode(input_ids, attention_mask=encoder_attention_mask)
        if decoder_input_ids is None:
            assert targets is not None, "decoder_input_ids are required when targets are not provided"
            decoder_input_ids = self._shift_right(targets)
            if decoder_attention_mask is None:
                decoder_attention_mask = self._shift_right_attention_mask(targets)

        logits, presents = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            if loss_reduction == "none":
                loss = loss.view_as(targets)
            return loss
        if use_cache:
            return logits, presents, encoder_hidden_states
        return logits

    @torch.inference_mode()
    def generate(self, input_tokens, max_tokens, decoder_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(input_tokens, list), "input_tokens must be a list of token ids"
        device = self.get_device()
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
        encoder_hidden_states = self.encode(input_ids)
        generated = [self.config.pad_token_id] if decoder_tokens is None else list(decoder_tokens)
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        past_key_values = None
        step_tokens = generated
        for _ in range(max_tokens):
            decoder_ids = torch.tensor([step_tokens], dtype=torch.long, device=device)
            logits, past_key_values, encoder_hidden_states = self.forward(
                decoder_input_ids=decoder_ids,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(next_logits, dim=-1, keepdim=True)
            token = next_ids.item()
            generated.append(token)
            yield token
            if token == self.config.eos_token_id:
                break
            step_tokens = [token]
