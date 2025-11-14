from collections.abc import Sequence
import torch
from torch import nn

try:
    from torchdrug import core, layers, models
    from torchdrug.core import Registry as R
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

    class GCN(nn.Module):
        def __init__(self, input_dim, hidden_dims, edge_input_dim=None,
                     short_cut=False, batch_norm=False, activation="relu",
                     concat_hidden=False, readout="sum"):
            super().__init__()
        def forward(self, graph, input, all_loss=None, metric=None):
            return {"node_feature": input, "graph_feature": input.mean(0, keepdim=True)}

    class RGCN(nn.Module):
        def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None,
                     short_cut=False, batch_norm=False, activation="relu",
                     concat_hidden=False, readout="sum"):
            super().__init__()
        def forward(self, graph, input, all_loss=None, metric=None):
            return {"node_feature": input, "graph_feature": input.mean(0, keepdim=True)}

    class GAT(nn.Module):
        def __init__(self, input_dim, hidden_dims, edge_input_dim=None,
                     num_head=1, negative_slope=0.2, short_cut=False,
                     batch_norm=False, activation="relu", concat_hidden=False,
                     readout="sum"):
            super().__init__()
        def forward(self, graph, input, all_loss=None, metric=None):
            return {"node_feature": input, "graph_feature": input.mean(0, keepdim=True)}

    class GIN(nn.Module):
        def __init__(self, input_dim, hidden_dims, edge_input_dim=None,
                     num_mlp_layer=2, eps=0, learn_eps=False, short_cut=False,
                     batch_norm=False, activation="relu", concat_hidden=False,
                     readout="sum"):
            super().__init__()
        def forward(self, graph, input, all_loss=None, metric=None):
            return {"node_feature": input, "graph_feature": input.mean(0, keepdim=True)}

    models = type('models', (), {
        'GCN': GCN, 'RGCN': RGCN, 'GAT': GAT, 'GIN': GIN
    })()

    class layers:
        class MaxReadout(nn.Module):
            def forward(self, graph, input):
                return input.max(0, keepdim=True)[0]
        class MeanReadout(nn.Module):
            def forward(self, graph, input):
                return input.mean(0, keepdim=True)
        class SumReadout(nn.Module):
            def forward(self, graph, input):
                return input.sum(0, keepdim=True)
        class MessagePassing(nn.Module):
            def __init__(self, input_dim, edge_input_dim, hidden_dims,
                         batch_norm=False, activation="relu"):
                super().__init__()
            def forward(self, graph, input):
                return input
        class Set2Set(nn.Module):
            def __init__(self, input_dim, num_step=3):
                super().__init__()
                self.output_dim = input_dim * 2
            def forward(self, graph, input):
                return input.repeat(1, 2)

from .readout import MeanReadout, SumReadout, MaxReadout, Set2Set


class CompGCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None,
                 batch_norm=False, activation="relu", composition="multiply"):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
    def forward(self, graph, input):
        output = self.linear(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        return self.activation(output)


@R.register("models.GlycanGCN")
class GlycanGCN(models.GCN):
    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None,
                 short_cut=False, batch_norm=False, activation="relu",
                 concat_hidden=False, readout="sum"):
        super(GlycanGCN, self).__init__(
            input_dim, hidden_dims, edge_input_dim, short_cut,
            batch_norm, activation, concat_hidden, readout.replace("dual", "mean")
        )
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout() if TORCHDRUG_AVAILABLE else MaxReadout()
            self.output_dim = self.output_dim * 2
    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGCN, self).forward(graph, input, all_loss, metric)
        if hasattr(self, "readout_ext"):
            f, g = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([g, self.readout_ext(graph, f)], dim=-1)
        return feature


@R.register("models.GlycanRGCN")
class GlycanRGCN(models.RGCN):
    def __init__(self, input_dim, hidden_dims, num_unit, num_relation,
                 edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanRGCN, self).__init__(
            input_dim, hidden_dims, num_relation, edge_input_dim,
            short_cut, batch_norm, activation, concat_hidden,
            readout.replace("dual", "mean")
        )
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout() if TORCHDRUG_AVAILABLE else MaxReadout()
            self.output_dim = self.output_dim * 2
    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanRGCN, self).forward(graph, input, all_loss, metric)
        if hasattr(self, "readout_ext"):
            f, g = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([g, self.readout_ext(graph, f)], dim=-1)
        return feature


@R.register("models.GlycanGAT")
class GlycanGAT(models.GAT):
    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None,
                 num_head=1, negative_slope=0.2, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGAT, self).__init__(
            input_dim, hidden_dims, edge_input_dim, num_head, negative_slope,
            short_cut, batch_norm, activation, concat_hidden,
            readout.replace("dual", "mean")
        )
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout() if TORCHDRUG_AVAILABLE else MaxReadout()
            self.output_dim = self.output_dim * 2
    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGAT, self).forward(graph, input, all_loss, metric)
        if hasattr(self, "readout_ext"):
            f, g = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([g, self.readout_ext(graph, f)], dim=-1)
        return feature


@R.register("models.GlycanGIN")
class GlycanGIN(models.GIN):
    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None,
                 num_mlp_layer=2, eps=0, learn_eps=False, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False,
                 readout="sum"):
        super(GlycanGIN, self).__init__(
            input_dim, hidden_dims, edge_input_dim, num_mlp_layer,
            eps, learn_eps, short_cut, batch_norm, activation,
            concat_hidden, readout.replace("dual", "mean")
        )
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout() if TORCHDRUG_AVAILABLE else MaxReadout()
            self.output_dim = self.output_dim * 2
    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGIN, self).forward(graph, input, all_loss, metric)
        if hasattr(self, "readout_ext"):
            f, g = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([g, self.readout_ext(graph, f)], dim=-1)
        return feature


@R.register("models.GlycanCompGCN")
class GlycanCompGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_relation, num_unit,
                 edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum",
                 composition="multiply"):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.embedding_init = nn.Embedding(num_unit, input_dim)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                CompGCNConv(self.dims[i], self.dims[i+1], num_relation,
                            edge_input_dim, batch_norm, activation, composition))
        if readout == "sum":
            self.readout = layers.SumReadout() if TORCHDRUG_AVAILABLE else SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout() if TORCHDRUG_AVAILABLE else MeanReadout()
        elif readout == "max":
            self.readout = layers.MaxReadout() if TORCHDRUG_AVAILABLE else MaxReadout()
        elif readout == "dual":
            if TORCHDRUG_AVAILABLE:
                self.readout1, self.readout2 = layers.MeanReadout(), layers.MaxReadout()
            else:
                self.readout1, self.readout2 = MeanReadout(), MaxReadout()
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden
        node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        if hasattr(self, "readout1"):
            graph_feature = torch.cat([self.readout1(graph, node_feature),
                                       self.readout2(graph, node_feature)], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)
        return {"graph_feature": graph_feature, "node_feature": node_feature}


@R.register("models.GlycanMPNN")
class GlycanMPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_unit, edge_input_dim,
                 num_layer=1, num_gru_layer=1, num_mlp_layer=2,
                 num_s2s_step=3, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False):
        super().__init__()
        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        feature_dim = hidden_dim * num_layer if concat_hidden else hidden_dim
        self.output_dim = feature_dim * 2
        self.node_output_dim = feature_dim
        self.num_layer = num_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.embedding_init = nn.Embedding(num_unit, hidden_dim)
        self.layer = layers.MessagePassing(hidden_dim, edge_input_dim,
                                           [hidden_dim] * (num_mlp_layer - 1),
                                           batch_norm, activation)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layer)
        self.readout = layers.Set2Set(feature_dim, num_step=num_s2s_step) if TORCHDRUG_AVAILABLE \
                       else Set2Set(feature_dim, num_step=num_s2s_step)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)
        hx = layer_input.repeat(self.gru.num_layers, 1, 1)
        for _ in range(self.num_layer):
            x = self.layer(graph, layer_input)
            hidden, hx = self.gru(x.unsqueeze(0), hx)
            hidden = hidden.squeeze(0)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden
        node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        graph_feature = self.readout(graph, node_feature)
        return {"graph_feature": graph_feature, "node_feature": node_feature}
