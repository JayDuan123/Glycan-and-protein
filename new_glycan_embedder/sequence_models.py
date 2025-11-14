import re
import math
import copy
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean

try:
    from torchdrug import core, layers, utils
    from torchdrug.core import Registry as R
    from torchdrug.layers import functional
    from torchdrug.utils import pretty
    TORCHDRUG_AVAILABLE = True
except ImportError:
    TORCHDRUG_AVAILABLE = False
    class Registry:
        _registry = {}
        @classmethod
        def register(cls, name):
            def decorator(func):
                cls._registry[name] = func
                return func
            return decorator
    R = Registry()

from .readout import MeanReadout, SumReadout, MaxReadout, AttentionReadout


def variadic_to_padded(input_tensor, sizes, value=0):
    max_size = sizes.max().item()
    batch_size = len(sizes)
    
    if input_tensor.dim() == 1:
        output = torch.full((batch_size, max_size),
                            value, dtype=input_tensor.dtype, device=input_tensor.device)
    else:
        output = torch.full((batch_size, max_size, input_tensor.size(-1)),
                            value, dtype=input_tensor.dtype, device=input_tensor.device)
    offset = 0
    for i, size in enumerate(sizes):
        output[i, :size] = input_tensor[offset:offset + size]
        offset += size
    mask = torch.arange(max_size, device=input_tensor.device).expand(batch_size, -1) < sizes.unsqueeze(1)
    return output, mask


def padded_to_variadic(padded_tensor, sizes):
    output_list = []
    for i, size in enumerate(sizes):
        output_list.append(padded_tensor[i, :size])
    return torch.cat(output_list, dim=0)


def _extend(bos, bos_size, input, input_size):
    extended_input = []
    extended_sizes = []
    
    bos_offset = 0
    input_offset = 0
    
    for i, (b_size, i_size) in enumerate(zip(bos_size, input_size)):
        batch_tokens = []
        
        if b_size > 0:
            batch_tokens.append(bos[bos_offset:bos_offset + b_size])
            bos_offset += b_size
            
        if i_size > 0:
            batch_tokens.append(input[input_offset:input_offset + i_size])
            input_offset += i_size
            
        if batch_tokens:
            extended_input.append(torch.cat(batch_tokens))
            extended_sizes.append(b_size + i_size)
        else:
            extended_sizes.append(0)
    
    if extended_input:
        final_input = torch.cat(extended_input)
    else:
        final_input = torch.empty(0, dtype=input.dtype, device=input.device)
        
    final_sizes = torch.tensor(extended_sizes, device=input.device)
    
    return final_input, final_sizes


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, input):
        batch_size, seq_len = input.shape[:2]
        position = torch.arange(seq_len, device=input.device).float()
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=input.device).float() *
                           -(math.log(10000.0) / self.dim))
        pos_emb = torch.zeros(seq_len, self.dim, device=input.device)
        pos_emb[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pos_emb[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        return pos_emb


class ProteinResNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation="gelu"):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation
        
        if input_dim != output_dim:
            self.shortcut = nn.Conv1d(input_dim, output_dim, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, mask=None):
        residual = self.shortcut(x.transpose(1, 2)).transpose(1, 2)
        
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        
        x = x + residual
        x = self.activation(x)
        
        if mask is not None:
            x = x * mask
            
        return x


class ProteinBERTBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, attention_dropout=0.1,
                 hidden_dropout=0.1, activation="gelu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            getattr(nn, activation.upper())() if activation.upper() in ['RELU', 'GELU'] else nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(hidden_dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        attn_mask = None
        if mask is not None:
            attn_mask = (mask.squeeze(-1) == 0)
            
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if mask is not None:
            x = x * mask
            
        return x


@R.register("models.GlycanConvolutionalNetwork")
class GlycanConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, glycoword_dim, kernel_size=3, stride=1, padding=1,
                activation='relu', short_cut=False, concat_hidden=False, readout="max"):
        super(GlycanConvolutionalNetwork, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i+1], kernel_size, stride, padding)
            )

        if readout == "sum":
            self.readout = SumReadout('glycan')
        elif readout == "mean":
            self.readout = MeanReadout('glycan')
        elif readout == "max":
            self.readout = MaxReadout('glycan')
        elif readout == "attention":
            self.readout = AttentionReadout(self.output_dim, 'glycan')
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input = variadic_to_padded(input, graph.num_glycowords, value=self.padding_id)[0]

        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        glycoword_feature = padded_to_variadic(hidden, graph.num_glycowords)
        graph_feature = self.readout(graph, glycoword_feature)
        
        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


@R.register("models.GlycanResNet")
class GlycanResNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[128, 128], glycoword_dim=216, kernel_size=5, stride=1, padding=2,
                 activation="gelu", short_cut=False, concat_hidden=False, layer_norm=False,
                 dropout=0, readout="max"):
        super(GlycanResNet, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.position_embedding = SinusoidalPositionEmbedding(hidden_dims[0])
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        self.layer_norm = nn.LayerNorm(hidden_dims[0]) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(ProteinResNetBlock(self.dims[i], self.dims[i + 1], kernel_size,
                                                         stride, padding, activation))

        if readout == "sum":
            self.readout = SumReadout("glycan")
        elif readout == "mean":
            self.readout = MeanReadout("glycan")
        elif readout == "attention":
            self.readout = AttentionReadout(self.output_dim, "glycan")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input, mask = variadic_to_padded(input, graph.num_glycowords, value=self.padding_id)
        mask = mask.unsqueeze(-1)

        input = self.embedding(input) + self.position_embedding(input).unsqueeze(0)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        input = input * mask
        
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input, mask)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        glycoword_feature = padded_to_variadic(hidden, graph.num_glycowords)
        graph_feature = self.readout(graph, glycoword_feature)
        
        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


@R.register("models.GlycanLSTM")
class GlycanLSTM(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=640, glycoword_dim=216, num_layers=3, activation='tanh', layer_norm=False,
                dropout=0):
        super(GlycanLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.node_output_dim = 2 * hidden_dim
        self.num_layers = num_layers
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=True)

        self.reweight = nn.Linear(2 * num_layers, 1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input = variadic_to_padded(input, graph.num_glycowords, value=self.padding_id)[0]
        
        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)

        output, hidden = self.lstm(input)

        glycoword_feature = padded_to_variadic(output, graph.num_glycowords)

        graph_feature = self.reweight(hidden[0].permute(1, 2, 0)).squeeze(-1)
        graph_feature = self.linear(graph_feature)
        graph_feature = self.activation(graph_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


@R.register("models.GlycanBERT")
class GlycanBERT(nn.Module):
    def __init__(self, input_dim=216, hidden_dim=512, num_layers=4, num_heads=8, intermediate_dim=2048,
                 activation="gelu", hidden_dropout=0.1, attention_dropout=0.1, max_position=8192):
        super(GlycanBERT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers

        self.num_glycoword_type = input_dim
        self.embedding = nn.Embedding(input_dim + 3, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(ProteinBERTBlock(hidden_dim, intermediate_dim, num_heads,
                                                       attention_dropout, hidden_dropout, activation))
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.glycoword_type
        size_ext = graph.num_glycowords

        bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.num_glycoword_type
        input, size_ext = _extend(bos, torch.ones_like(size_ext), input, size_ext)

        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * (self.num_glycoword_type + 1)
        input, size_ext = _extend(input, size_ext, eos, torch.ones_like(size_ext))

        input, mask = variadic_to_padded(input, size_ext, value=self.num_glycoword_type + 2)
        mask = mask.long().unsqueeze(-1)

        input = self.embedding(input)
        position_indices = torch.arange(input.shape[1], device=input.device)
        input = input + self.position_embedding(position_indices).unsqueeze(0)
        input = self.layer_norm(input)
        input = self.dropout(input)

        for layer in self.layers:
            input = layer(input, mask)

        glycoword_feature = padded_to_variadic(input, graph.num_glycowords)

        graph_feature = input[:, 0]
        graph_feature = self.linear(graph_feature)
        graph_feature = F.tanh(graph_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }
