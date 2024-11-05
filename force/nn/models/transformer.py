import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from force.nn import ConfigurableModule
from force.nn.layers import NAMED_POINTWISE_ACTIVATIONS


def scaled_dot_product_attention(
        query, key, value,
        attn_mask=None, dropout_p=0.0, need_weights=True,
        training=True
):
    logits = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if attn_mask is not None:
        assert attn_mask.dtype is torch.bool
        logits.masked_fill_(attn_mask, float('-inf'))
    attn_weight = torch.softmax(logits, dim=-1)
    if attn_mask is not None:
        attn_weight = attn_weight.masked_fill(attn_mask, 0.0)
    attn_weight = F.dropout(attn_weight, p=dropout_p, training=training)
    return attn_weight @ value, attn_weight if need_weights else None


class Attention(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        self.WQ = nn.Parameter(torch.empty(dim, dim))
        self.WK = nn.Parameter(torch.empty(dim, dim))
        self.WV = nn.Parameter(torch.empty(dim, dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in [self.WQ, self.WK, self.WV]:
            nn.init.xavier_uniform_(p)

    def merge_masks(self, attn_mask, key_padding_mask, query):
        merged_mask = None

        if key_padding_mask is not None:
            merged_mask = key_padding_mask

        if attn_mask is not None:
            batch_size, seq_len, _ = query.shape

            if attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, seq_len, seq_len).expand(batch_size, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, seq_len)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None
        return merged_mask

    def forward(self, query, key, value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):
        Q = query @ self.WQ
        K = key @ self.WK
        V = value @ self.WV
        merged_mask = self.merge_masks(attn_mask, key_padding_mask, query)
        return scaled_dot_product_attention(
            Q, K, V,
            attn_mask=merged_mask,
            dropout_p=self.dropout,
            need_weights=need_weights,
            training=self.training
        )


# Decoder-only layer attends only to inputs (no outputs from encoder)
class DecoderOnlyTransformerLayer(ConfigurableModule):
    class Config(ConfigurableModule.Config):
        dim_feedforward = 2048
        num_heads = 1
        dropout = 0.1
        activation = 'relu'
        norm_first = False

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, cfg, dim_model, layer_norm_eps):
        assert cfg.num_heads == 1, 'Currently only a single attention head is supported'
        super().__init__(cfg)
        self.dim_model = dim_model
        # self.attention = nn.MultiheadAttention(dim_model, cfg.num_heads,
        #                                        dropout=cfg.dropout, batch_first=True)
        self.attention = Attention(dim_model, cfg.dropout)

        # Feedforward model
        self.linear1 = nn.Linear(dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, dim_model)

        self.norm_first = cfg.norm_first
        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.activation = NAMED_POINTWISE_ACTIVATIONS[cfg.activation]

    def get_output_shape(self, shape, **kwargs):
        assert isinstance(shape, torch.Size) and len(shape) == 2
        assert shape[1] == self.dim_model
        return shape

    def forward(self, input, attn_mask=None, key_padding_mask=None):
        x = input
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Stack of decoder-only layers
class DecoderOnlyTransformer(ConfigurableModule):
    class Config(ConfigurableModule.Config):
        dim_model = 128
        layer_norm_eps = 1e-5
        layer = DecoderOnlyTransformerLayer.Config
        num_layers = 3
        use_final_layer_norm = True

    def __init__(self, cfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            DecoderOnlyTransformerLayer(cfg.layer, cfg.dim_model, cfg.layer_norm_eps)
            for _ in range(cfg.num_layers)
        ])

        if cfg.use_final_layer_norm:
            self.norm = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)
        else:
            self.norm = None

    def get_output_shape(self, shape, **kwargs):
        assert isinstance(shape, torch.Size) and len(shape) == 2
        assert shape[-1] == self.cfg.dim_model
        return shape

    def forward(self, input, **kwargs):
        B, T, D = input.shape
        assert D == self.cfg.dim_model
        output = input
        for layer in self.layers:
            output = layer(output, **kwargs)
        if self.norm is not None:
            output = self.norm(output)
        return output