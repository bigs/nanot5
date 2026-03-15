"""
Minimal T5 generation engine with the repo's chat/tool sentinel behavior.
"""

import signal
import warnings
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr):
    expr = expr.replace(",", "")
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed_chars for x in expr):
        return None

    dangerous_patterns = [
        "__", "import", "exec", "eval", "compile", "open", "file", "input",
        "raw_input", "globals", "locals", "vars", "dir", "getattr", "setattr",
        "delattr", "hasattr",
    ]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None
    if ".count(" not in expr:
        return None
    return eval_with_timeout(expr)


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        probs = F.softmax(vals / temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


class KVCache:
    """
    Minimal FA3-style KV cache helper retained for attention fallback tests.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        self.cache_seqlens.zero_()

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens


class RowState:
    def __init__(self, prompt_tokens):
        self.current_tokens = prompt_tokens.copy()
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _encode_prompt(self, tokens, num_samples):
        device = self.model.get_device()
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        encoder_hidden_states = self.model.encode(input_ids, attention_mask=attention_mask)
        if num_samples == 1:
            return encoder_hidden_states, attention_mask
        encoder_hidden_states = encoder_hidden_states.expand(num_samples, -1, -1).contiguous()
        attention_mask = attention_mask.expand(num_samples, -1).contiguous()
        return encoder_hidden_states, attention_mask

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list) and all(isinstance(token, int) for token in tokens), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        python_start = self.tokenizer.get_chat_token_id("python_start")
        python_end = self.tokenizer.get_chat_token_id("python_end")
        output_start = self.tokenizer.get_chat_token_id("output_start")
        output_end = self.tokenizer.get_chat_token_id("output_end")
        assistant_end = self.tokenizer.get_chat_token_id("assistant_end")
        eos = self.tokenizer.get_eos_token_id()
        document_start = self.tokenizer.get_document_start_token_id()
        decoder_start = self.tokenizer.get_decoder_start_token_id()

        encoder_hidden_states, encoder_attention_mask = self._encode_prompt(tokens, num_samples)
        decoder_input_ids = torch.full((num_samples, 1), decoder_start, dtype=torch.long, device=device)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.bool)
        past_key_values = None
        row_states = [RowState(tokens) for _ in range(num_samples)]
        max_tokens = self.model.config.decoder_sequence_len if max_tokens is None else max_tokens

        num_generated = 0
        while num_generated < max_tokens and not all(state.completed for state in row_states):
            logits, past_key_values, encoder_hidden_states = self.model.forward(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            sampled_tokens = sample_next_token(logits[:, -1, :], rng, temperature, top_k)[:, 0].tolist()

            token_column = []
            token_masks = []
            for row_idx, state in enumerate(row_states):
                if state.completed:
                    token_masks.append(0)
                    token_column.append(document_start)
                    continue

                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[row_idx]
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                if next_token in {assistant_end, eos, document_start}:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(self.tokenizer.encode(str(result)))
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1
            decoder_input_ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.bool)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.get_chat_token_id("assistant_end")
        eos = self.tokenizer.get_eos_token_id()
        document_start = self.tokenizer.get_document_start_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for row_idx, (token, mask) in enumerate(zip(token_column, token_masks)):
                if completed[row_idx]:
                    continue
                if token in {assistant_end, eos, document_start}:
                    completed[row_idx] = True
                    continue
                results[row_idx].append(token)
                masks[row_idx].append(mask)
            if all(completed):
                break
        return results, masks
