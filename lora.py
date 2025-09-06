from typing import Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRAParam(nn.Module):
    def __init__(self, weight: torch.Tensor, r: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        out_features, in_features = weight.shape
        self.r = r
        self.scaling = alpha / float(r) if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if self.r == 0 or self.scaling == 0.0:
            return W
        return W + self.scaling * (self.B @ self.A)


def inject_lora_into_lstm(lstm: nn.LSTM, r: int, alpha: float, dropout: float, targets: Tuple[str, ...] = ("ih", "hh")) -> None:
    L = lstm.num_layers
    for l in range(L):
        if "ih" in targets:
            base = getattr(lstm, f"weight_ih_l{l}")
            parametrize.register_parametrization(module=lstm,
                                                 tensor_name=f"weight_ih_l{l}",
                                                 parametrization=LoRAParam(base.data,
                                                                           r=r,
                                                                           alpha=alpha,
                                                                           dropout=dropout), )
            base.requires_grad = False
        if "hh" in targets:
            base = getattr(lstm, f"weight_hh_l{l}")
            parametrize.register_parametrization(module=lstm,
                                                 tensor_name=f"weight_hh_l{l}",
                                                 parametrization=LoRAParam(base.data,
                                                                           r=r,
                                                                           alpha=alpha,
                                                                           dropout=dropout), )
            base.requires_grad = False


def inject_lora_into_linear(linear: nn.Linear, r: int, alpha: float, dropout: float) -> None:
    base = linear.weight
    parametrize.register_parametrization(module=linear,
                                         tensor_name="weight",
                                         parametrization=LoRAParam(base.data,
                                                                   r=r,
                                                                   alpha=alpha,
                                                                   dropout=dropout))
    base.requires_grad = False


def merge_all_lora(module: nn.Module) -> None:
    for m in module.modules():
        try:
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        except Exception:
            pass
        # Handle LSTM named params
        if isinstance(m, nn.LSTM):
            for l in range(m.num_layers):
                for kind in ("ih", "hh"):
                    tname = f"weight_{kind}_l{l}"
                    try:
                        parametrize.remove_parametrizations(m, tname, leave_parametrized=True)
                    except Exception:
                        pass


def mark_trainable_lora_only(model: nn.Module, train_bias: bool = True, train_layernorm: bool = True):
    total, trainable = 0, 0
    for n, p in model.named_parameters():
        total += p.numel()
        is_lora = (".A" in n) or (".B" in n) or ("lora" in n.lower())
        is_bias = n.endswith("bias")
        is_ln = ("norm" in n.lower()) or ("layernorm" in n.lower())
        ok = False
        if is_lora:
            ok = True
        elif train_bias and is_bias:
            ok = True
        elif train_layernorm and is_ln:
            ok = True
        p.requires_grad = ok
        if ok:
            trainable += p.numel()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
