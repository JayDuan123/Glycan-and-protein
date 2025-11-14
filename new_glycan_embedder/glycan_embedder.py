import os
import re
import numpy as np
from typing import List, Optional, Dict, Union
import logging

import torch
import torch.nn as nn

from .glycan_data import Glycan, PackedGlycan, GlycanParser
from .glycan_vocab import VocabularyManager
from .sequence_models import GlycanConvolutionalNetwork, GlycanResNet, GlycanLSTM, GlycanBERT
from .graph_models import GlycanGCN, GlycanRGCN, GlycanGAT, GlycanGIN, GlycanCompGCN, GlycanMPNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycanEmbedder:
    EMBEDDER_CLASSES = {
        'gcn': GlycanGCN,
        'rgcn': GlycanRGCN,
        'gat': GlycanGAT,
        'gin': GlycanGIN,
        'compgcn': GlycanCompGCN,
        'mpnn': GlycanMPNN,
        'cnn': GlycanConvolutionalNetwork,
        'resnet': GlycanResNet,
        'lstm': GlycanLSTM,
        'bert': GlycanBERT
    }

    GRAPH_METHODS = ['gcn', 'rgcn', 'gat', 'gin', 'compgcn', 'mpnn']
    SEQUENCE_METHODS = ['cnn', 'resnet', 'lstm', 'bert']
    ALL_METHODS = GRAPH_METHODS + SEQUENCE_METHODS

    def __init__(self, vocab_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.vocab_manager = VocabularyManager(vocab_path)
        self.parser = GlycanParser(self.vocab_manager)
        self._embedders = {}

    def _create_embedder(self, method: str, embedding_dim: int, **kwargs):
        method = method.lower()
        if method not in self.ALL_METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {self.ALL_METHODS}")

        embedder_class = self.EMBEDDER_CLASSES[method]
        vocab_sizes = self.vocab_manager.get_vocab_sizes()

        if method in self.GRAPH_METHODS:
            params = {
                'input_dim': embedding_dim,
                'hidden_dims': kwargs.get('hidden_dims', [embedding_dim]),
                'num_unit': vocab_sizes['num_units']
            }

            if method in ['rgcn', 'compgcn']:
                params['num_relation'] = vocab_sizes['num_links']
            elif method == 'mpnn':
                params['hidden_dim'] = embedding_dim
                params['edge_input_dim'] = kwargs.get('edge_input_dim', 84)
                params.pop('hidden_dims')
            elif method == 'gat':
                params['num_head'] = kwargs.get('num_head', 4)
                params['negative_slope'] = kwargs.get('negative_slope', 0.2)
            elif method == 'gin':
                params['num_mlp_layer'] = kwargs.get('num_mlp_layer', 2)
                params['eps'] = kwargs.get('eps', 0)
                params['learn_eps'] = kwargs.get('learn_eps', False)
            elif method == 'compgcn':
                params['composition'] = kwargs.get('composition', 'multiply')

        else:
            params = {
                'input_dim': kwargs.get('input_dim', embedding_dim),
                'glycoword_dim': vocab_sizes['num_glycowords']
            }

            if method == 'bert':
                params['hidden_dim'] = embedding_dim
                params['num_layers'] = kwargs.get('num_layers', 8)
                params['num_heads'] = kwargs.get('num_heads', 12)
                params['intermediate_dim'] = kwargs.get('intermediate_dim', 3072)
            elif method == 'lstm':
                params['hidden_dim'] = embedding_dim
                params['num_layers'] = kwargs.get('num_layers', 3)
            elif method in ['cnn', 'resnet']:
                params['hidden_dims'] = kwargs.get('hidden_dims', [embedding_dim])

        common_params = [
            'short_cut', 'batch_norm', 'activation', 'concat_hidden', 'readout',
            'layer_norm', 'dropout', 'kernel_size', 'stride', 'padding'
        ]

        for param in common_params:
            if param in kwargs:
                params[param] = kwargs[param]

        return embedder_class(**params).to(self.device)

    def embed_glycans(self, glycan_list: List[str], method: str = 'gcn',
                      embedding_dim: int = 128, **kwargs) -> torch.Tensor:
        method = method.lower()

        embedder_key = f"{method}_{embedding_dim}_{hash(str(sorted(kwargs.items())))}"
        if embedder_key not in self._embedders:
            self._embedders[embedder_key] = self._create_embedder(
                method, embedding_dim, **kwargs
            )

        embedder = self._embedders[embedder_key]

        glycans = []
        for iupac in glycan_list:
            try:
                glycan = self.parser.from_iupac(iupac, device=self.device)
                glycans.append(glycan)
            except Exception as e:
                logger.warning(f"Failed to parse glycan '{iupac}': {e}")
                empty_unit_type = torch.tensor([0], device=self.device)
                empty_glycoword_type = torch.tensor([0], device=self.device)
                empty_glycan = Glycan(
                    unit_type=empty_unit_type,
                    glycoword_type=empty_glycoword_type,
                    vocab_manager=self.vocab_manager,
                    device=self.device
                )
                glycans.append(empty_glycan)

        packed_glycan = PackedGlycan(glycans)

        with torch.no_grad():
            output = embedder(packed_glycan, input=None, all_loss=None, metric=None)
            embeddings = output["graph_feature"]

        return embeddings

    def get_available_methods(self) -> Dict[str, List[str]]:
        return {
            'graph_based': self.GRAPH_METHODS,
            'sequence_based': self.SEQUENCE_METHODS,
            'all': self.ALL_METHODS
        }


def embed_glycans(glycan_list: List[str], method: str = 'gcn',
                  embedding_dim: int = 128, vocab_path: Optional[str] = None,
                  device: str = 'cpu', **kwargs) -> np.ndarray:
    embedder = GlycanEmbedder(vocab_path, device)
    embeddings = embedder.embed_glycans(glycan_list, method, embedding_dim, **kwargs)
    return embeddings.cpu().numpy()


def get_available_methods() -> Dict[str, List[str]]:
    return {
        'graph_based': GlycanEmbedder.GRAPH_METHODS,
        'sequence_based': GlycanEmbedder.SEQUENCE_METHODS,
        'all': GlycanEmbedder.ALL_METHODS
    }


__all__ = [
    'GlycanEmbedder',
    'embed_glycans',
    'get_available_methods',
    'GlycanGCN',
    'GlycanRGCN',
    'GlycanGAT',
    'GlycanGIN',
    'GlycanCompGCN',
    'GlycanMPNN',
    'GlycanConvolutionalNetwork',
    'GlycanResNet',
    'GlycanLSTM',
    'GlycanBERT'
]


if __name__ == "__main__":
    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
        "Neu5Ac(a2-3)Gal(b1-4)Glc"
    ]

    print("Available methods:")
    for category, methods in get_available_methods().items():
        print(f"  {category}: {methods}")

    for method in ['gcn', 'lstm', 'bert', 'rgcn', 'mpnn']:
        try:
            embeds = embed_glycans(glycans, method=method, embedding_dim=128, readout='dual')
            print(f"{method.upper()} embeddings shape: {embeds.shape}")
        except Exception as e:
            print(f"{method.upper()} failed: {e}")
