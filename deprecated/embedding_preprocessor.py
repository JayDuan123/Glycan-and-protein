"""
Embedding Preprocessor - Precomputes and caches glycan and protein embeddings
Prevents redundant embedding computation during data loading
"""
import os
import numpy as np
import pandas as pd
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch

from Protein_Sequence_Embedder import ProteinEmbedderFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPreprocessor:
    """
    Preprocessor for glycan and protein embeddings with intelligent caching
    """

    def __init__(self,
                 cache_dir: str = "preprocessed_embeddings",
                 protein_model: str = "650M",
                 protein_model_dir: str = "resources/esm-model-weights",
                 glycan_method: str = "lstm",
                 glycan_vocab_path: Optional[str] = None,
                 glycan_hidden_dims: Optional[List[int]] = None,
                 glycan_readout: str = "mean",
                 device: Optional[str] = None):
        """
        Initialize the preprocessor

        Args:
            cache_dir: Directory to store cached embeddings
            protein_model: ESM2 model size
            protein_model_dir: Path to protein model weights
            glycan_method: Glycan embedding method
            glycan_vocab_path: Path to glycan vocabulary
            glycan_hidden_dims: Hidden dimensions for glycan embedder
            glycan_readout: Readout function for graph-based methods
            device: Computing device
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.protein_cache_dir = self.cache_dir / "proteins"
        self.glycan_cache_dir = self.cache_dir / "glycans"
        self.protein_cache_dir.mkdir(exist_ok=True)
        self.glycan_cache_dir.mkdir(exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'mps' # Only for macOS MPS devices

        # Initialize embedders
        logger.info("Initializing protein embedder...")
        self.protein_embedder = ProteinEmbedderFactory.create_embedder(
            "esm2",
            model_name=protein_model,
            model_dir=protein_model_dir,
            device=self.device
        )

        # Store glycan embedder parameters for lazy initialization
        self.glycan_params = {
            'method': glycan_method,
            'vocab_path': glycan_vocab_path,
            'hidden_dims': glycan_hidden_dims,
            'readout': glycan_readout,
            'device': self.device
        }

        # Lazy initialize glycan embedder when needed
        self._glycan_embedder = None

        logger.info(f"Preprocessor initialized with cache dir: {self.cache_dir}")

    @property
    def glycan_embedder(self):
        """Lazy initialization of glycan embedder"""
        if self._glycan_embedder is None:
            from Integrated_Embedder import GlycanProteinPairEmbedder
            logger.info("Initializing glycan embedder...")

            # Create a temporary pair embedder to access glycan embedder
            temp_embedder = GlycanProteinPairEmbedder(
                protein_model=self.protein_embedder.model_key.split('_')[2],  # Extract model size
                glycan_method=self.glycan_params['method'],
                glycan_vocab_path=self.glycan_params['vocab_path'],
                glycan_hidden_dims=self.glycan_params['hidden_dims'],
                glycan_readout=self.glycan_params['readout'],
                device=self.device
            )
            self._glycan_embedder = temp_embedder.glycan_embedder

        return self._glycan_embedder

    def _get_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_protein_cache_path(self, sequence: str) -> Path:
        """Get cache file path for protein"""
        hash_key = self._get_hash(sequence)
        return self.protein_cache_dir / f"{hash_key}.npy"

    def _get_glycan_cache_path(self, iupac: str) -> Path:
        """Get cache file path for glycan"""
        hash_key = self._get_hash(iupac)
        return self.glycan_cache_dir / f"{hash_key}.npy"

    def precompute_protein_embeddings(self,
                                      sequences: List[str],
                                      batch_size: int = 32,
                                      force_recompute: bool = False) -> Dict[str, str]:
        """
        Precompute protein embeddings and return sequence -> cache_path mapping

        Args:
            sequences: List of protein sequences
            batch_size: Batch size for embedding computation
            force_recompute: Whether to recompute existing embeddings

        Returns:
            Dictionary mapping sequences to cache file paths
        """
        unique_sequences = list(set(sequences))
        sequence_to_cache = {}
        sequences_to_compute = []

        logger.info(f"Processing {len(unique_sequences)} unique protein sequences...")

        # Check which embeddings need to be computed
        for sequence in unique_sequences:
            cache_path = self._get_protein_cache_path(sequence)
            sequence_to_cache[sequence] = str(cache_path)

            if not cache_path.exists() or force_recompute:
                sequences_to_compute.append(sequence)

        if sequences_to_compute:
            logger.info(f"Computing {len(sequences_to_compute)} protein embeddings...")

            # Compute embeddings in batches
            for i in tqdm(range(0, len(sequences_to_compute), batch_size),
                          desc="Computing protein embeddings"):
                batch_sequences = sequences_to_compute[i:i + batch_size]

                # Compute embeddings (memory optimized)
                embeddings = self.protein_embedder.embed(batch_sequences)

                # Save each embedding
                for j, sequence in enumerate(batch_sequences):
                    cache_path = self._get_protein_cache_path(sequence)
                    np.save(cache_path, embeddings[j])

                # Memory cleanup
                del embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"Protein embedding preprocessing complete. Cache: {self.protein_cache_dir}")
        return sequence_to_cache

    def precompute_glycan_embeddings(self,
                                     iupacs: List[str],
                                     batch_size: int = 32,
                                     force_recompute: bool = False) -> Dict[str, str]:
        """
        Precompute glycan embeddings and return IUPAC -> cache_path mapping

        Args:
            iupacs: List of IUPAC glycan sequences
            batch_size: Batch size for embedding computation
            force_recompute: Whether to recompute existing embeddings

        Returns:
            Dictionary mapping IUPAC sequences to cache file paths
        """
        unique_iupacs = list(set(iupacs))
        iupac_to_cache = {}
        iupacs_to_compute = []

        logger.info(f"Processing {len(unique_iupacs)} unique glycan IUPAC sequences...")

        # Check which embeddings need to be computed
        for iupac in unique_iupacs:
            cache_path = self._get_glycan_cache_path(iupac)
            iupac_to_cache[iupac] = str(cache_path)

            if not cache_path.exists() or force_recompute:
                iupacs_to_compute.append(iupac)

        if iupacs_to_compute:
            logger.info(f"Computing {len(iupacs_to_compute)} glycan embeddings...")

            # Prepare glycan embedder parameters
            embedding_dim = self.protein_embedder.get_embedding_dim()
            glycan_params = {
                'method': self.glycan_params['method'],
                'embedding_dim': embedding_dim
            }

            if self.glycan_params['hidden_dims']:
                glycan_params['hidden_dims'] = self.glycan_params['hidden_dims']

            if self.glycan_params['method'] in ['gcn', 'rgcn', 'gat', 'gin', 'compgcn', 'mpnn']:
                glycan_params['readout'] = self.glycan_params['readout']

            # Compute embeddings in batches
            for i in tqdm(range(0, len(iupacs_to_compute), batch_size),
                          desc="Computing glycan embeddings"):
                batch_iupacs = iupacs_to_compute[i:i + batch_size]

                # Compute embeddings (memory optimized)
                with torch.no_grad():
                    embeddings = self.glycan_embedder.embed_glycans(
                        batch_iupacs,
                        **glycan_params
                    )

                    # Convert to numpy if tensor
                    if torch.is_tensor(embeddings):
                        embeddings = embeddings.cpu().numpy()

                # Save each embedding
                for j, iupac in enumerate(batch_iupacs):
                    cache_path = self._get_glycan_cache_path(iupac)
                    np.save(cache_path, embeddings[j])

                # Memory cleanup
                del embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"Glycan embedding preprocessing complete. Cache: {self.glycan_cache_dir}")
        return iupac_to_cache

    def preprocess_dataset(self,
                           data_path: str,
                           protein_col: str = 'target',
                           sequence_col: str = 'target',
                           exclude_cols: Optional[List[str]] = None,
                           batch_size: int = 32,
                           force_recompute: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Preprocess entire dataset - extract unique sequences and precompute embeddings

        Args:
            data_path: Path to dataset CSV/Excel file
            protein_col: Column name for protein identifiers
            sequence_col: Column name for protein sequences
            exclude_cols: Columns to exclude from glycan analysis
            batch_size: Batch size for embedding computation
            force_recompute: Whether to recompute existing embeddings

        Returns:
            Tuple of (protein_cache_mapping, glycan_cache_mapping)
        """
        logger.info(f"Preprocessing dataset: {data_path}")

        # Load data
        if str(data_path).endswith('.csv'):
            data = pd.read_csv(data_path, engine='python')
        elif str(data_path).endswith(('.xlsx', '.xls')):
            data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Get unique protein sequences
        data = data.dropna(subset=[sequence_col])
        unique_proteins = data[sequence_col].unique().tolist()

        # Get glycan columns (numeric columns only)
        exclude_set = {protein_col, sequence_col} | set(exclude_cols or [])
        if 'protein' not in exclude_set:
            exclude_set.add('protein')

        potential_glycan_cols = [col for col in data.columns if col not in exclude_set]
        glycan_cols = []
        for col in potential_glycan_cols:
            try:
                pd.to_numeric(data[col], errors='raise')
                glycan_cols.append(col)
            except (ValueError, TypeError):
                continue

        unique_glycans = glycan_cols  # Assuming column names are IUPAC sequences

        logger.info(f"Found {len(unique_proteins)} unique proteins and {len(unique_glycans)} unique glycans")

        # Precompute embeddings
        protein_cache_mapping = self.precompute_protein_embeddings(
            unique_proteins, batch_size=batch_size, force_recompute=force_recompute
        )

        glycan_cache_mapping = self.precompute_glycan_embeddings(
            unique_glycans, batch_size=batch_size, force_recompute=force_recompute
        )

        # Save mappings for future use
        mapping_file = self.cache_dir / "cache_mappings.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump({
                'protein_mapping': protein_cache_mapping,
                'glycan_mapping': glycan_cache_mapping,
                'data_path': data_path,
                'glycan_params': self.glycan_params
            }, f)

        logger.info(f"Preprocessing complete. Mappings saved to: {mapping_file}")
        return protein_cache_mapping, glycan_cache_mapping

    def load_cached_embeddings(self,
                               sequences: List[str],
                               cache_mapping: Dict[str, str]) -> np.ndarray:
        """
        Load cached embeddings for given sequences

        Args:
            sequences: List of sequences (proteins or glycans)
            cache_mapping: Mapping from sequence to cache file path

        Returns:
            Array of embeddings
        """
        embeddings = []
        for sequence in sequences:
            if sequence not in cache_mapping:
                raise ValueError(f"No cached embedding found for sequence: {sequence[:50]}...")

            cache_path = cache_mapping[sequence]
            if not os.path.exists(cache_path):
                raise ValueError(f"Cache file not found: {cache_path}")

            embedding = np.load(cache_path)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_cache_info(self) -> Dict:
        """Get information about the embedding cache"""
        protein_files = list(self.protein_cache_dir.glob("*.npy"))
        glycan_files = list(self.glycan_cache_dir.glob("*.npy"))

        protein_size = sum(f.stat().st_size for f in protein_files)
        glycan_size = sum(f.stat().st_size for f in glycan_files)

        return {
            'protein_embeddings': len(protein_files),
            'glycan_embeddings': len(glycan_files),
            'protein_cache_size_mb': protein_size / (1024 * 1024),
            'glycan_cache_size_mb': glycan_size / (1024 * 1024),
            'total_cache_size_mb': (protein_size + glycan_size) / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

    def clear_cache(self, cache_type: str = "all"):
        """
        Clear embedding cache

        Args:
            cache_type: "all", "proteins", or "glycans"
        """
        import shutil

        if cache_type in ["all", "proteins"]:
            if self.protein_cache_dir.exists():
                shutil.rmtree(self.protein_cache_dir)
                self.protein_cache_dir.mkdir(exist_ok=True)
                logger.info("Cleared protein embedding cache")

        if cache_type in ["all", "glycans"]:
            if self.glycan_cache_dir.exists():
                shutil.rmtree(self.glycan_cache_dir)
                self.glycan_cache_dir.mkdir(exist_ok=True)
                logger.info("Cleared glycan embedding cache")

        if cache_type == "all":
            mapping_file = self.cache_dir / "cache_mappings.pkl"
            if mapping_file.exists():
                mapping_file.unlink()
                logger.info("Cleared cache mappings")


# Convenience function
def preprocess_embeddings(data_path: str,
                          cache_dir: str = "preprocessed_embeddings",
                          protein_model: str = "650M",
                          glycan_method: str = "lstm",
                          glycan_vocab_path: Optional[str] = None,
                          batch_size: int = 32,
                          force_recompute: bool = False,
                          **kwargs) -> EmbeddingPreprocessor:
    """
    Convenience function to preprocess embeddings for a dataset

    Args:
        data_path: Path to dataset file
        cache_dir: Directory for caching embeddings
        protein_model: ESM2 model size
        glycan_method: Glycan embedding method
        glycan_vocab_path: Path to glycan vocabulary
        batch_size: Batch size for computation
        force_recompute: Whether to recompute existing embeddings
        **kwargs: Additional arguments for preprocessor

    Returns:
        Initialized preprocessor with computed embeddings
    """
    preprocessor = EmbeddingPreprocessor(
        cache_dir=cache_dir,
        protein_model=protein_model,
        glycan_method=glycan_method,
        glycan_vocab_path=glycan_vocab_path,
        **kwargs
    )

    preprocessor.preprocess_dataset(
        data_path=data_path,
        batch_size=batch_size,
        force_recompute=force_recompute
    )

    return preprocessor


if __name__ == "__main__":
    # Example usage
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    data_path = "data/v12_glycan_binding.csv"

    # Preprocess embeddings
    preprocessor = preprocess_embeddings(
        data_path=data_path,
        cache_dir="preprocessed_embeddings",
        protein_model="650M",
        glycan_method="lstm",
        glycan_vocab_path=vocab_path,
        batch_size=12
    )

    # Show cache info
    cache_info = preprocessor.get_cache_info()
    print(f"Cache info: {cache_info}")