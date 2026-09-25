"""Flan-T5-Base encoder, used exactly as Supra2-IMG's inference.py uses it -- frozen by
default, optionally with its top blocks trainable.

Supra: tokenizer("google/flan-t5-base"), max_length 128, truncation, fp32 weights run
under bf16 autocast, `last_hidden_state`, and the empty string "" encoded as the
unconditional context for CFG. The only difference is padding="longest" instead of
padding="max_length": T5 masks padded keys, so real-token outputs are unchanged and
typical batches (FLUX-Reason captions: p50 ~48, p90 ~70 tokens) run ~2x cheaper.

train_blocks=N unfreezes the top N of the 12 encoder blocks plus the final layer norm.
The lower blocks (and the relative-position bias, which lives in block 0) stay frozen, so
general language knowledge is kept while the top re-shapes caption embeddings for the
image model. Dropout stays off (eval mode) either way.
"""
from __future__ import annotations

import torch


def unfreeze_top_blocks(t5_encoder_model, n: int) -> list:
    """Freeze everything, then unfreeze the top `n` encoder blocks and the final layer norm;
    n < 0 unfreezes the WHOLE encoder (token embeddings and relative-position bias too).
    Returns the trainable (name, parameter) pairs."""
    t5_encoder_model.requires_grad_(n < 0)
    if n > 0:
        enc = t5_encoder_model.encoder
        if not 0 < n <= len(enc.block):
            raise ValueError(f"train_blocks={n}, but the encoder has {len(enc.block)} blocks")
        for blk in enc.block[-n:]:
            blk.requires_grad_(True)
        enc.final_layer_norm.requires_grad_(True)
    return [(name, p) for name, p in t5_encoder_model.named_parameters() if p.requires_grad]


class T5TextEncoder:
    def __init__(self, path: str, device, max_len: int = 128, train_blocks: int = 0):
        from transformers import AutoTokenizer, T5EncoderModel
        self.device, self.max_len = torch.device(device), max_len
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = T5EncoderModel.from_pretrained(path, torch_dtype=torch.float32).to(self.device)
        self.model.eval().requires_grad_(False)
        self.dim = self.model.config.d_model
        # (name, parameter) of everything that trains; empty when frozen
        self.trainable = unfreeze_top_blocks(self.model, train_blocks)

    def tokenize(self, captions: list[str], pad_to: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU only -- safe to run on the prefetch thread. `pad_to` rounds the length up
        (masked padding) so a compiled model sees few distinct shapes."""
        t = self.tok(captions, padding="longest", truncation=True, max_length=self.max_len,
                     return_tensors="pt")
        ids, mask = t["input_ids"], t["attention_mask"]
        extra = (-ids.shape[1]) % pad_to
        if extra:
            ids = torch.nn.functional.pad(ids, (0, extra), value=self.tok.pad_token_id)
            mask = torch.nn.functional.pad(mask, (0, extra))
        return ids, mask

    def encode(self, ids: torch.Tensor, mask: torch.Tensor, grad: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = ids.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
        with torch.set_grad_enabled(grad and bool(self.trainable)), \
                torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return h.float(), mask

    def __call__(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ctx (B, L, 768) fp32, mask (B, L) long). Never builds a graph."""
        return self.encode(*self.tokenize(captions))


def _load_modernbert(path: str):
    """AutoModel.from_pretrained, or -- where transformers refuses a .bin checkpoint on
    torch < 2.6 (Ettin ships only pytorch_model.bin) -- build from the config and load the
    state dict with weights_only=True. The MLM head's tensors are dropped."""
    from transformers import AutoConfig, AutoModel
    try:
        return AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    except ValueError:
        import os
        m = AutoModel.from_config(AutoConfig.from_pretrained(path))
        f = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else None
        if f is None:
            from huggingface_hub import hf_hub_download
            f = hf_hub_download(path, "pytorch_model.bin")
        sd = {k.removeprefix("model."): v for k, v in torch.load(f, map_location="cpu", weights_only=True).items()}
        missing, _ = m.load_state_dict(sd, strict=False)
        if missing:
            raise KeyError(f"{path}: encoder tensors missing from the checkpoint: {missing[:5]}")
        return m


class EttinTextEncoder:
    """Ettin (jhu-clsp/ettin-encoder-68m, MIT): a ModernBERT encoder, 68M, hidden 512,
    byte-level BPE (keeps case, punctuation, accents and non-Latin brand names that
    Flan-T5's SentencePiece maps to <unk>). Same interface as T5TextEncoder.

    train_blocks != 0 trains the WHOLE encoder (there is no "top blocks" option: it is
    small, and the conditioning space has to be learned anew anyway). The empty string
    encodes as [CLS][SEP], the unconditional context for CFG."""

    def __init__(self, path: str, device, max_len: int = 128, train_blocks: int = 0, grad_ckpt: bool = False):
        from transformers import AutoTokenizer
        self.device, self.max_len = torch.device(device), max_len
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = _load_modernbert(path).to(self.device)
        self.model.eval().requires_grad_(bool(train_blocks))
        if grad_ckpt and train_blocks:
            # HF only checkpoints in train() mode; every Ettin dropout is 0.0 (config), so train()
            # changes nothing numerically. Recomputes each layer in the backward: trainable Ettin
            # activations were ~22 GB/GPU at 128 x 416 tokens uncompiled (2026-09-25 OOM).
            if any(getattr(self.model.config, k, 0) for k in ("attention_dropout", "mlp_dropout", "embedding_dropout")):
                raise ValueError("Ettin checkpointing needs train() mode, but this config has dropout")
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()
        self.dim = self.model.config.hidden_size
        self.trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def tokenize(self, captions: list[str], pad_to: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tok(captions, padding="longest", truncation=True, max_length=self.max_len,
                     return_tensors="pt")
        ids, mask = t["input_ids"], t["attention_mask"]
        extra = (-ids.shape[1]) % pad_to
        if extra:
            ids = torch.nn.functional.pad(ids, (0, extra), value=self.tok.pad_token_id)
            mask = torch.nn.functional.pad(mask, (0, extra))
        return ids, mask

    def encode(self, ids: torch.Tensor, mask: torch.Tensor, grad: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = ids.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
        with torch.set_grad_enabled(grad and bool(self.trainable)), \
                torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return h.float(), mask

    def __call__(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(*self.tokenize(captions))


def build_text_encoder(kind: str, path: str, device, max_len: int = 128, train_blocks: int = 0,
                       grad_ckpt: bool = False):
    if kind == "t5":
        return T5TextEncoder(path, device, max_len=max_len, train_blocks=train_blocks)
    if kind == "ettin":
        return EttinTextEncoder(path, device, max_len=max_len, train_blocks=train_blocks, grad_ckpt=grad_ckpt)
    raise ValueError(f"unknown text encoder {kind!r} (expected t5 or ettin)")


class HashTextEncoder:
    """Stand-in for tests and --debug_tiny: deterministic per-character embeddings, same
    interface and masking behaviour, no download. "" gives one token, like T5's </s>.
    train_blocks > 0 makes the embedding table trainable (the whole "encoder")."""

    def __init__(self, dim: int = 768, device="cpu", max_len: int = 128, seed: int = 0,
                 train_blocks: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(257, dim, generator=g), requires_grad=bool(train_blocks))
        self.device, self.max_len, self.dim = torch.device(device), max_len, dim
        self.trainable = [("table", self.table)] if train_blocks else []

    def tokenize(self, captions: list[str], pad_to: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [[256] + [b for b in c.encode("utf-8")][: self.max_len - 1] for c in captions]
        L = max(len(i) for i in ids)
        L += (-L) % pad_to
        mask = torch.zeros(len(ids), L, dtype=torch.long)
        idx = torch.zeros(len(ids), L, dtype=torch.long)
        for r, i in enumerate(ids):
            idx[r, : len(i)] = torch.tensor(i)
            mask[r, : len(i)] = 1
        return idx, mask

    def encode(self, ids: torch.Tensor, mask: torch.Tensor, grad: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad and bool(self.trainable)):
            return self.table[ids.cpu()].to(self.device), mask.to(self.device)

    def __call__(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(*self.tokenize(captions))
