import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import logging
from pathlib import Path
import urllib.request
import re
from torchdrug.transforms import TruncateProtein
from torchdrug.data import Protein

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinEmbedder(ABC):
    @abstractmethod
    def embed(self, sequences: Union[str, List[str]]) -> np.ndarray:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass


class ESM2Embedder(ProteinEmbedder):
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

    VALID_AA_CHARS = set('AFCUDNEQGHLIKOMPRSTVWY')

    def __init__(self, model_name: str = "650M", model_dir: str = "./models",
                 device: Optional[str] = None, repr_layer: int = -1):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.truncate_transform = TruncateProtein(max_length=1022, random=False)
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.model_key = None
        for key, info in self.MODELS.items():
            if info["alias"] == model_name:
                self.model_key = key
                break

        if not self.model_key:
            raise ValueError(f"Model {model_name} not found. Available: 650M, 3B")

        self.model_info = self.MODELS[self.model_key]
        self.repr_layer = repr_layer if repr_layer != -1 else self.model_info["layers"]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        self._load_model()

    def _truncate(self, sequence: str) -> str:
        protein = Protein.from_sequence(sequence)
        sample = {"graph": protein}
        truncate_transform = TruncateProtein(max_length=1022, random=False)
        sample_truncated = truncate_transform(sample)
        protein_truncated = sample_truncated["graph"]
        sequence = protein_truncated.to_sequence()
        return sequence

    def _clean_protein_sequence(self, sequence: str) -> str:
        if not sequence:
            return sequence
        sequence = sequence.upper()
        unrecognized_chars = set(sequence) - self.VALID_AA_CHARS
        if unrecognized_chars:
            cleaned_sequence = sequence
            for char in unrecognized_chars:
                cleaned_sequence = cleaned_sequence.replace(char, 'G')
            short_cleaned_sequence = self._truncate(cleaned_sequence)
            return short_cleaned_sequence
        else:
            short_seq = self._truncate(sequence)
        return short_seq

    def _clean_sequences(self, sequences: List[str]) -> List[str]:
        cleaned_sequences = []
        for i, seq in enumerate(sequences):
            cleaned_seq = self._clean_protein_sequence(seq)
            cleaned_sequences.append(cleaned_seq)
        return cleaned_sequences

    def _download_model(self):
        model_path = self.checkpoints_dir / f"{self.model_key}.pt"
        if model_path.exists():
            logger.info(f"Model already exists at {model_path}")
            return model_path
        logger.info(f"Downloading {self.model_key} to {model_path}...")
        url = self.model_info["url"]
        logger.info(f"Downloading from {url}")
        temp_path = model_path.with_suffix('.tmp')

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rDownload progress: {percent:.1f}%", end='')

        try:
            urllib.request.urlretrieve(url, temp_path, reporthook=download_progress)
            print()
            temp_path.rename(model_path)
            logger.info(f"Model downloaded and saved to {model_path}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
        return model_path

    def _load_model(self):
        model_path = self._download_model()
        try:
            import esm
            import torch.hub
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")

        original_hub_dir = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(self.model_dir))
            logger.info(f"Loading model architecture and weights from {model_path}...")
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_key)
        finally:
            torch.hub.set_dir(original_hub_dir)

        self.model = model.to(self.device)
        self.model.eval()
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        logger.info(f"Loaded {self.model_key} with embedding dimension {self.get_embedding_dim()}")

    def get_embedding_dim(self) -> int:
        return self.model_info["embedding_dim"]

    def embed(self, sequences: Union[str, List[str]]) -> np.ndarray:
        if isinstance(sequences, str):
            sequences = [sequences]
        cleaned_sequences = self._clean_sequences(sequences)
        data = [(f"seq_{i}", seq) for i, seq in enumerate(cleaned_sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
            embeddings = results["representations"][self.repr_layer]
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sequence_embeddings = []
            for i, (_, seq) in enumerate(data):
                seq_len = len(seq)
                seq_embedding = embeddings[i, 1:seq_len+1].mean(0)
                sequence_embeddings.append(seq_embedding)

            embeddings_tensor = torch.stack(sequence_embeddings)
            del embeddings, sequence_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result_numpy = embeddings_tensor.cpu().numpy()
            del embeddings_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del batch_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result_numpy


class ProteinEmbedderFactory:
    _embedders = {
        "esm2": ESM2Embedder
    }

    @classmethod
    def create_embedder(cls, embedder_type: str, **kwargs) -> ProteinEmbedder:
        if embedder_type not in cls._embedders:
            raise ValueError(f"Unknown embedder type: {embedder_type}. Available: {list(cls._embedders.keys())}")
        return cls._embedders[embedder_type](**kwargs)

    @classmethod
    def register_embedder(cls, name: str, embedder_class: type):
        if not issubclass(embedder_class, ProteinEmbedder):
            raise ValueError(f"{embedder_class} must inherit from ProteinEmbedder")
        cls._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name}")


def embed_proteins(sequences: Union[str, List[str]],
                  model: str = "650M",
                  embedder_type: str = "esm2",
                  **kwargs) -> np.ndarray:
        embedder = ProteinEmbedderFactory.create_embedder(
            embedder_type,
            model_name=model,
            **kwargs
        )
        return embedder.embed(sequences)


if __name__ == "__main__":
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KAL*ARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "SEQUENCE[WITH]BRACKETS*AND*STARS"
    ]

    print("Using ESM-2 650M model with sequence cleaning (Memory Optimized):")
    embeddings_650m = embed_proteins(
        sequences,
        model="650M",
        model_dir="../resources/esm-model-weights"
    )
    print(f"Embeddings shape: {embeddings_650m.shape}")
    print(f"First sequence embedding (first 10 dims): {embeddings_650m[0, :10]}")
