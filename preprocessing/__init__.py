"""
Preprocessing module for glycan-protein data preparation.

This module provides utilities for preprocessing glycan and protein data,
including embedding preprocessing and clustering-based data splitting.
"""

from embedding_preprocessor import EmbeddingPreprocessor, preprocess_embeddings
from clustering_splitter import ProteinClusteringSplitter

# Define what should be imported with "from preprocessing import *"
__all__ = [
    'EmbeddingPreprocessor',
    'preprocess_embeddings',
    'ProteinClusteringSplitter',
]
