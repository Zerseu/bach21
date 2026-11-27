"""The neural network: token embedding, one LSTM layer, optional causal self-attention, output layer."""
import torch
from torch.nn import Module

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CausalSelfAttention(Module):
    """Multi-head self-attention where each position only sees itself and earlier positions,
    followed by a residual connection and LayerNorm."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(d_model)
        self.register_buffer('_mask', None, persistent=False)

    def _causal_mask(self, length: int, device) -> torch.Tensor:
        if self._mask is None or self._mask.size(0) != length:
            self._mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
        return self._mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._causal_mask(x.size(1), x.device)
        y, _ = self.mha(x, x, x, attn_mask=mask)
        return self.ln(x + y)


class TorchModule(Module):
    """Next-token model. ``forward`` maps token ids (batch, time) to logits (batch, time, vocabulary);
    the logits at position t predict token t + 1.

    The attribute names below are the keys of saved model files, so renaming them breaks loading
    existing ``*_model.torch`` files.
    """

    def __init__(self, vocabulary_size: int, map_direct: dict[str, int], map_reverse: dict[int, str], hidden_size: int,
                 lora_enable: bool = False, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_targets: str = 'ih,hh,out,emb', add_causal_attn: bool = True, attn_heads: int = 4,
                 dropout: float = 0.25, output_dropout: float = 0.0, tie_weights: bool = False):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.map_direct = map_direct
        self.map_reverse = map_reverse
        self.hidden_size = hidden_size

        # The creation order matters: it decides how the seeded random generator initialises the weights.
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_dropout = torch.nn.Dropout(p=output_dropout)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=vocabulary_size)
        if tie_weights:
            # Input and output embeddings share one matrix: fewer weights, and rare tokens learn from both ends.
            torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.linear.weight = self.embedding.weight

        self.emb_post = None
        self.post_attn = CausalSelfAttention(hidden_size, n_heads=attn_heads) if add_causal_attn else None

        if lora_enable:
            self.apply_lora(lora_r, lora_alpha, lora_dropout, lora_targets)

        self.to(DEVICE)

    def apply_lora(self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                   lora_targets: str = 'ih,hh,out,emb'):
        """Wraps the current weights with LoRA adapters. Call it after loading a trained base model when
        fine-tuning, since the wrapped base weights are frozen.

        lora_targets is a comma-separated subset of: ih, hh (LSTM input and recurrent weights),
        out (output layer), emb (adds an adapted identity projection after the embedding).
        """
        from lora import inject_lora_into_linear, inject_lora_into_lstm
        targets = tuple(x.strip() for x in lora_targets.split(',') if x.strip())
        if 'emb' in targets:
            self.emb_post = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            torch.nn.init.eye_(self.emb_post.weight)
            inject_lora_into_linear(self.emb_post, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        if 'ih' in targets or 'hh' in targets:
            inject_lora_into_lstm(self.lstm, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                                  targets=tuple(x for x in ('ih', 'hh') if x in targets))
        if 'out' in targets:
            inject_lora_into_linear(self.linear, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.to(DEVICE)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = self.embedding(x)
        if self.emb_post is not None:
            x = self.emb_post(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        if self.post_attn is not None:
            x = self.post_attn(x)
        x = self.output_dropout(x)
        return self.linear(x)
