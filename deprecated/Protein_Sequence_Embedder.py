"""
Protein Sequence Embedder using ESM-2 models
MEMORY OPTIMIZED VERSION - Prevents GPU memory accumulation
"""
import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import logging
from pathlib import Path
import urllib.request
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinEmbedder(ABC):
    """Abstract base class for protein sequence embedders"""

    @abstractmethod
    def embed(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """
        Embed protein sequences into vectors

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            np.ndarray: Embedding matrix (n_sequences x embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings"""
        pass


class ESM2Embedder(ProteinEmbedder):
    """ESM-2 model embedder for protein sequences
    MEMORY OPTIMIZED VERSION - Prevents GPU memory accumulation during embedding"""

    MODELS = {
        "esm2_t33_650M_UR50D": {
            "embedding_dim": 1280,
            "layers": 33,
            "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
            "alias": "650M"
        },
        "esm2_t36_3B_UR50D": {
            "embedding_dim": 2560,
            "layers": 36,
            "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
            "alias": "3B"
        }
    }

    # Valid amino acid characters recognized by ESM models
    VALID_AA_CHARS = set('ACDEFGHIKLMNPQRSTVWYXUOBZJ')

    def __init__(self, model_name: str = "650M", model_dir: str = "./models",
                 device: Optional[str] = None, repr_layer: int = -1):
        """
        Initialize ESM-2 embedder

        Args:
            model_name: Model size - "650M" or "3B"
            model_dir: Directory to store downloaded models
            device: Device to run model on (cuda/cpu). Auto-detected if None
            repr_layer: Which layer to extract representations from (-1 for last)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Create checkpoints subdirectory (where ESM expects models)
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Find model by alias
        self.model_key = None
        for key, info in self.MODELS.items():
            if info["alias"] == model_name:
                self.model_key = key
                break

        if not self.model_key:
            raise ValueError(f"Model {model_name} not found. Available: 650M, 3B")

        self.model_info = self.MODELS[self.model_key]
        self.repr_layer = repr_layer if repr_layer != -1 else self.model_info["layers"]

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model()

    def _clean_protein_sequence(self, sequence: str) -> str:
        """
        Clean protein sequence by replacing unrecognized characters with glycine (G)

        Args:
            sequence: Raw protein sequence

        Returns:
            str: Cleaned sequence with only valid amino acid characters
        """
        if not sequence:
            return sequence

        # Convert to uppercase
        sequence = sequence.upper()

        # Find unrecognized characters
        unrecognized_chars = set(sequence) - self.VALID_AA_CHARS

        if unrecognized_chars:
            logger.warning(f"Found unrecognized characters in sequence: {unrecognized_chars}")
            logger.warning(f"Replacing with glycine (G): {', '.join(sorted(unrecognized_chars))}")

            # Replace each unrecognized character with G
            cleaned_sequence = sequence
            for char in unrecognized_chars:
                cleaned_sequence = cleaned_sequence.replace(char, 'G')

            return cleaned_sequence

        return sequence

    def _clean_sequences(self, sequences: List[str]) -> List[str]:
        """
        Clean a list of protein sequences

        Args:
            sequences: List of raw protein sequences

        Returns:
            List[str]: List of cleaned sequences
        """
        cleaned_sequences = []
        for i, seq in enumerate(sequences):
            cleaned_seq = self._clean_protein_sequence(seq)
            cleaned_sequences.append(cleaned_seq)

            # Log if sequence was modified
            if cleaned_seq != seq.upper():
                logger.info(f"Sequence {i} modified: '{seq[:50]}...' -> '{cleaned_seq[:50]}...'")

        return cleaned_sequences

    def _download_model(self):
        """Download model if not present"""
        # ESM expects models in checkpoints subdirectory
        model_path = self.checkpoints_dir / f"{self.model_key}.pt"

        if model_path.exists():
            logger.info(f"Model already exists at {model_path}")
            return model_path

        logger.info(f"Downloading {self.model_key} to {model_path}...")

        url = self.model_info["url"]
        logger.info(f"Downloading from {url}")

        # Create temporary file path
        temp_path = model_path.with_suffix('.tmp')

        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rDownload progress: {percent:.1f}%", end='')

        try:
            urllib.request.urlretrieve(url, temp_path, reporthook=download_progress)
            print()  # New line after progress

            # Rename temp file to final name
            temp_path.rename(model_path)
            logger.info(f"Model downloaded and saved to {model_path}")

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if temp_path.exists():
                temp_path.unlink()  # Clean up partial download
            raise

        return model_path

    def _load_model(self):
        """Load the ESM model"""
        # Ensure model is downloaded
        model_path = self._download_model()

        try:
            import esm
            import torch.hub
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")

        # Save original hub dir
        original_hub_dir = torch.hub.get_dir()

        try:
            # Temporarily set hub dir to our model dir
            # This tells ESM where to look for the model
            torch.hub.set_dir(str(self.model_dir))

            # Now load_model_and_alphabet will find our downloaded file
            # without trying to download it again
            logger.info(f"Loading model architecture and weights from {model_path}...")
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_key)

        finally:
            # Always restore original hub dir
            torch.hub.set_dir(original_hub_dir)

        self.model = model.to(self.device)
        self.model.eval()
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()

        logger.info(f"Loaded {self.model_key} with embedding dimension {self.get_embedding_dim()}")

    def get_embedding_dim(self) -> int:
        """Get embedding dimension for the model"""
        return self.model_info["embedding_dim"]

    def embed(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """
        Embed protein sequences
        MEMORY OPTIMIZED VERSION - Prevents GPU memory accumulation

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            np.ndarray: Embeddings of shape (n_sequences, embedding_dim)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Clean sequences to remove unrecognized characters
        logger.debug(f"Cleaning {len(sequences)} sequences...")
        cleaned_sequences = self._clean_sequences(sequences)

        # Prepare batch
        data = [(f"seq_{i}", seq) for i, seq in enumerate(cleaned_sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Get embeddings with memory optimization
        with torch.no_grad():
            # 🔧 MEMORY FIX 1: ESM模型计算
            results = self.model(batch_tokens, repr_layers=[self.repr_layer],
                               return_contacts=False)
            embeddings = results["representations"][self.repr_layer]

            # 🔧 MEMORY FIX 2: 立即清理results以释放GPU内存
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Average over sequence length (excluding special tokens)
            sequence_embeddings = []
            for i, (_, seq) in enumerate(data):
                # Get sequence length (excluding BOS/EOS tokens)
                seq_len = len(seq)
                # Extract embeddings for actual sequence (position 1 to seq_len+1)
                seq_embedding = embeddings[i, 1:seq_len+1].mean(0)
                sequence_embeddings.append(seq_embedding)

            embeddings_tensor = torch.stack(sequence_embeddings)

            # 🔧 MEMORY FIX 3: 清理中间变量
            del embeddings, sequence_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 🔧 MEMORY FIX 4: 立即转换为CPU numpy并清理GPU tensor
            result_numpy = embeddings_tensor.cpu().numpy()
            del embeddings_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 🔧 MEMORY FIX 5: 清理batch tokens
        del batch_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result_numpy


class ProteinEmbedderFactory:
    """Factory class for creating protein embedders"""

    _embedders = {
        "esm2": ESM2Embedder
    }

    @classmethod
    def create_embedder(cls, embedder_type: str, **kwargs) -> ProteinEmbedder:
        """
        Create a protein embedder

        Args:
            embedder_type: Type of embedder (e.g., "esm2")
            **kwargs: Arguments for the specific embedder

        Returns:
            ProteinEmbedder instance
        """
        if embedder_type not in cls._embedders:
            raise ValueError(f"Unknown embedder type: {embedder_type}. "
                           f"Available: {list(cls._embedders.keys())}")

        return cls._embedders[embedder_type](**kwargs)

    @classmethod
    def register_embedder(cls, name: str, embedder_class: type):
        """
        Register a new embedder type

        Args:
            name: Name for the embedder type
            embedder_class: Class that implements ProteinEmbedder
        """
        if not issubclass(embedder_class, ProteinEmbedder):
            raise ValueError(f"{embedder_class} must inherit from ProteinEmbedder")

        cls._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name}")


# Convenience function
def embed_proteins(sequences: Union[str, List[str]],
                  model: str = "650M",
                  embedder_type: str = "esm2",
                  **kwargs) -> np.ndarray:
    """
    Convenience function to embed protein sequences
    MEMORY OPTIMIZED VERSION

    Args:
        sequences: Protein sequence(s) to embed
        model: Model size for ESM2 ("650M" or "3B")
        embedder_type: Type of embedder to use
        **kwargs: Additional arguments for the embedder

    Returns:
        np.ndarray: Embedding matrix
    """
    embedder = ProteinEmbedderFactory.create_embedder(
        embedder_type,
        model_name=model,
        **kwargs
    )
    return embedder.embed(sequences)


if __name__ == "__main__":
    # Example usage with sequences containing unrecognized characters
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KAL*ARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",  # Contains *
        "SEQUENCE[WITH]BRACKETS*AND*STARS",  # Contains [], *
    ]

    # Using 650M model with custom model directory
    print("Using ESM-2 650M model with sequence cleaning (Memory Optimized):")
    embeddings_650m = embed_proteins(
        sequences,
        model="650M",
        model_dir="../resources/esm-model-weights"  # Custom model directory
    )
    print(f"Embeddings shape: {embeddings_650m.shape}")
    print(f"First sequence embedding (first 10 dims): {embeddings_650m[0, :10]}")

    # Using 3B model (uncomment to test - requires more memory)
    # print("\nUsing ESM-2 3B model:")
    # embeddings_3b = embed_proteins(sequences, model="3B", model_dir="resources/esm-model-weights")
    # print(f"Embeddings shape: {embeddings_3b.shape}")