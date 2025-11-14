import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path
import pickle
import hashlib
import os
import time

from embedding_preprocessor import EmbeddingPreprocessor
from clustering_splitter import ProteinClusteringSplitter, create_clustered_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecomputedGlycanProteinDataset(Dataset):
    def __init__(self,
                 pairs: List[Tuple[str, str]],
                 targets: torch.Tensor,
                 protein_cache_mapping: Optional[Dict[str, str]] = None,
                 glycan_cache_mapping: Optional[Dict[str, str]] = None,
                 pair_cache_mapping: Optional[Dict[str, str]] = None,
                 fusion_method: str = "concat",
                 device: Optional[str] = None):
        self.pairs = pairs
        self.targets = targets.cpu()
        self.protein_cache_mapping = protein_cache_mapping
        self.glycan_cache_mapping = glycan_cache_mapping
        self.pair_cache_mapping = pair_cache_mapping
        self.fusion_method = fusion_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_pre_fused = pair_cache_mapping is not None
        
        if self.use_pre_fused:
            logger.info(f"Using pre-fused embeddings (fusion method: {fusion_method})")
        else:
            if protein_cache_mapping is None or glycan_cache_mapping is None:
                raise ValueError("Either pair_cache_mapping or both protein_cache_mapping and glycan_cache_mapping must be provided")
            if fusion_method != "concat":
                raise ValueError(
                    f"Runtime fusion only supports 'concat', got '{fusion_method}'. "
                    f"For attention fusion, use pre-fused embeddings."
                )
            logger.info(f"Using individual embeddings with runtime concat fusion")

        assert len(self.pairs) == len(self.targets)
        logger.info(f"Initialized precomputed dataset: {len(self.pairs)} samples")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        glycan_iupac, protein_sequence = self.pairs[idx]

        if self.use_pre_fused:
            pair_key = f"{glycan_iupac}|{protein_sequence}"
            if pair_key not in self.pair_cache_mapping:
                raise ValueError(f"No cached pair embedding for: {pair_key}")
            pair_path = self.pair_cache_mapping[pair_key]
            combined_emb = torch.FloatTensor(np.load(pair_path))
        else:
            if protein_sequence not in self.protein_cache_mapping:
                raise ValueError(f"No cached protein embedding")
            if glycan_iupac not in self.glycan_cache_mapping:
                raise ValueError(f"No cached glycan embedding")

            protein_emb = torch.FloatTensor(np.load(self.protein_cache_mapping[protein_sequence]))
            glycan_emb = torch.FloatTensor(np.load(self.glycan_cache_mapping[glycan_iupac]))

            protein_emb = nn.functional.normalize(protein_emb, p=2, dim=0)
            glycan_emb = nn.functional.normalize(glycan_emb, p=2, dim=0)
            combined_emb = torch.cat([glycan_emb, protein_emb], dim=0)

        return combined_emb, self.targets[idx]


class OptimizedEnhancedGlycanProteinDataLoader:
    def __init__(self,
                 data_path: str = "data/v12_glycan_binding.csv",
                 embedder=None,
                 protein_col: str = 'target',
                 sequence_col: str = 'target',
                 exclude_cols: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 cache_dir: str = "optimized_embeddings",
                 use_precomputed: bool = True,
                 use_clustering: bool = True,
                 clustering_params: Optional[Dict] = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.data_path = data_path
        self.protein_col = protein_col
        self.sequence_col = sequence_col
        self.cache_dir = Path(cache_dir)
        self.use_precomputed = use_precomputed
        self.use_clustering = use_clustering
        self.exclude_cols = exclude_cols or []
        if 'protein' not in self.exclude_cols:
            self.exclude_cols.append('protein')

        self.clustering_params = clustering_params or {
            'n_clusters': 10,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'sequence_identity': 0.5,
            'coverage': 0.8,
            'use_pca': True
        }

        self.preprocessor = None
        self.splitter = None
        self.protein_splits = None
        self.protein_cache_mapping = None
        self.glycan_cache_mapping = None
        self.pair_cache_mapping = None

        self.timing_info = {
            'data_loading': 0,
            'clustering': 0,
            'splitting': 0,
            'embedding_computation': 0,
            'total_setup': 0
        }

        start_time = time.time()
        self.data = self._load_data(data_path)
        self.glycan_columns = self._get_glycan_columns()
        self.timing_info['data_loading'] = time.time() - start_time

        logger.info(f"Loaded data: {len(self.data)} samples, {len(self.glycan_columns)} glycans")
        logger.info(f"Using OPTIMIZED workflow")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        try:
            if str(data_path).endswith('.csv'):
                data = pd.read_csv(data_path, engine='python')
            elif str(data_path).endswith('.xlsx'):
                data = pd.read_excel(data_path, engine='openpyxl')
            elif str(data_path).endswith('.xls'):
                data = pd.read_excel(data_path, engine='xlrd')
            else:
                raise ValueError(f"Unsupported file format")

            data = data.dropna(subset=[self.sequence_col])

            potential_glycan_cols = [
                col for col in data.columns
                if col not in {self.protein_col, self.sequence_col} | set(self.exclude_cols)
            ]

            glycan_cols = []
            for col in potential_glycan_cols:
                try:
                    pd.to_numeric(data[col], errors='raise')
                    glycan_cols.append(col)
                except:
                    pass

            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _get_glycan_columns(self) -> List[str]:
        exclude_set = {self.protein_col, self.sequence_col} | set(self.exclude_cols)
        potential_glycan_cols = [col for col in self.data.columns if col not in exclude_set]
        glycan_cols = []
        for col in potential_glycan_cols:
            try:
                pd.to_numeric(self.data[col], errors='raise')
                glycan_cols.append(col)
            except:
                pass
        return glycan_cols

    def setup_optimized_workflow(self,
                             protein_model: str = "650M",
                             glycan_method: str = "lstm",
                             glycan_vocab_path: Optional[str] = None,
                             glycan_hidden_dims: Optional[List[int]] = None,
                             glycan_readout: str = "mean",
                             fusion_method: str = "concat",
                             force_recompute: bool = False,
                             save_splitter_path: Optional[str] = None,
                             **kwargs):
        total_start = time.time()
        logger.info("Starting OPTIMIZED workflow")

        clustering_start = time.time()
        unique_proteins = self.data[self.sequence_col].unique().tolist()
        mock_embeddings = np.zeros((len(unique_proteins), 100))
        
        self.protein_splits, self.splitter = create_clustered_splits(
            protein_embeddings=mock_embeddings,
            protein_sequences=unique_proteins,
            save_splitter_path=save_splitter_path,
            **self.clustering_params
        )
        
        self.timing_info['clustering'] = time.time() - clustering_start

        splitting_start = time.time()
        all_required_pairs = set()
        for split_name, split_proteins in self.protein_splits.items():
            for protein in split_proteins:
                protein_data = self.data[self.data[self.sequence_col] == protein]
                for _, row in protein_data.iterrows():
                    for glycan in self.glycan_columns:
                        if not pd.isna(row[glycan]):
                            all_required_pairs.add((glycan, protein))
        
        self.timing_info['splitting'] = time.time() - splitting_start

        embedding_start = time.time()
        if self.use_precomputed:
            self.preprocessor = EmbeddingPreprocessor(
                cache_dir=str(self.cache_dir),
                protein_model=protein_model,
                glycan_method=glycan_method,
                glycan_vocab_path=glycan_vocab_path,
                glycan_hidden_dims=glycan_hidden_dims,
                glycan_readout=glycan_readout,
                fusion_method=fusion_method,
                device=self.device,
                **kwargs
            )

            self.pair_cache_mapping = self.preprocessor.precompute_fused_pair_embeddings(
                list(all_required_pairs),
                batch_size=64,
                force_recompute=force_recompute
            )
            
            self.protein_cache_mapping = None
            self.glycan_cache_mapping = None

            mapping_file = self.cache_dir / "optimized_cache_mappings.pkl"
            with open(mapping_file, 'wb') as f:
                pickle.dump({
                    'pair_cache_mapping': self.pair_cache_mapping,
                    'protein_splits': self.protein_splits,
                    'clustering_params': self.clustering_params,
                    'fusion_method': fusion_method,
                    'embedding_params': {
                        'protein_model': protein_model,
                        'glycan_method': glycan_method,
                        'glycan_vocab_path': glycan_vocab_path,
                        'glycan_hidden_dims': glycan_hidden_dims,
                        'glycan_readout': glycan_readout
                    },
                    'optimization_stats': {
                        'total_pairs': len(all_required_pairs),
                        'timing_info': self.timing_info
                    }
                }, f)

        self.timing_info['embedding_computation'] = time.time() - embedding_start
        self.timing_info['total_setup'] = time.time() - total_start
        logger.info("OPTIMIZED workflow completed")

    def create_pairs_dataset(self,
                             glycan_subset: Optional[List[str]] = None,
                             protein_subset: Optional[List[str]] = None,
                             max_pairs: Optional[int] = None) -> Tuple[List[Tuple[str, str]], List[float]]:
        data = self.data.copy()
        if protein_subset:
            data = data[data[self.protein_col].isin(protein_subset)]
        glycans_to_use = glycan_subset if glycan_subset else self.glycan_columns
        pairs = []
        strengths = []

        for _, row in data.iterrows():
            protein_sequence = row[self.sequence_col]
            for glycan in glycans_to_use:
                if glycan in data.columns:
                    value = row[glycan]
                    if pd.isna(value):
                        continue
                    pairs.append((glycan, protein_sequence))
                    strengths.append(float(value))

        if max_pairs and len(pairs) > max_pairs:
            idx = np.random.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in idx]
            strengths = [strengths[i] for i in idx]

        return pairs, strengths

    def create_pytorch_dataloader(self,
                                  protein_subset: Optional[List[str]] = None,
                                  glycan_subset: Optional[List[str]] = None,
                                  batch_size: int = 32,
                                  shuffle: bool = True,
                                  num_workers: int = 0,
                                  normalize_targets: bool = True,
                                  max_pairs: Optional[int] = None,
                                  fusion_method: str = "concat",
                                  **kwargs):

        if protein_subset and self.protein_cache_mapping:
            missing = set(protein_subset) - set(self.protein_cache_mapping.keys())
            if missing:
                additional = self.preprocessor.precompute_protein_embeddings(
                    list(missing), force_recompute=False
                )
                self.protein_cache_mapping.update(additional)

        pairs, strengths = self.create_pairs_dataset(
            glycan_subset=glycan_subset,
            protein_subset=protein_subset,
            max_pairs=max_pairs
        )

        strengths_tensor = torch.FloatTensor(strengths)
        target_scaler = None

        if normalize_targets:
            from sklearn.preprocessing import StandardScaler
            target_scaler = StandardScaler()
            strengths_norm = target_scaler.fit_transform(
                strengths_tensor.numpy().reshape(-1, 1)
            ).flatten()
            strengths_tensor = torch.FloatTensor(strengths_norm)

        if self.use_precomputed and self.pair_cache_mapping:
            dataset = PrecomputedGlycanProteinDataset(
                pairs=pairs,
                targets=strengths_tensor,
                pair_cache_mapping=self.pair_cache_mapping,
                fusion_method=fusion_method,
                device=self.device
            )
        else:
            raise NotImplementedError("Only pre-fused embeddings supported")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(self.device.startswith("cuda"))
        )

        return dataloader, target_scaler

    def create_train_val_test_loaders(self,
                                      batch_size: int = 32,
                                      normalize_targets: bool = True,
                                      max_pairs_per_split: Optional[int] = None,
                                      fusion_method: str = "concat",
                                      protein_model: str = "650M",
                                      glycan_method: str = "lstm",
                                      glycan_vocab_path: Optional[str] = None,
                                      glycan_hidden_dims: Optional[List[int]] = None,
                                      glycan_readout: str = "mean",
                                      force_recompute: bool = False,
                                      save_splitter_path: Optional[str] = None,
                                      plot_analysis: bool = False,
                                      **kwargs):

        if self.protein_splits is None:
            self.setup_optimized_workflow(
                protein_model=protein_model,
                glycan_method=glycan_method,
                glycan_vocab_path=glycan_vocab_path,
                glycan_hidden_dims=glycan_hidden_dims,
                glycan_readout=glycan_readout,
                fusion_method=fusion_method,
                force_recompute=force_recompute,
                save_splitter_path=save_splitter_path
            )

        dataloaders = {}
        scalers = {}

        for split_name, proteins in self.protein_splits.items():
            if len(proteins) == 0:
                continue

            loader, scaler = self.create_pytorch_dataloader(
                protein_subset=proteins,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                normalize_targets=normalize_targets,
                max_pairs=max_pairs_per_split,
                fusion_method=fusion_method,
                **kwargs
            )

            dataloaders[split_name] = loader
            scalers[split_name] = scaler

        dataloaders['target_scaler'] = scalers.get('train')
        return dataloaders

    def get_cache_info(self) -> Dict:
        if self.preprocessor:
            info = self.preprocessor.get_cache_info()
            info['optimization_enabled'] = True
            info['timing_info'] = self.timing_info
            return info
        else:
            return {
                "optimization_enabled": False,
                "cache_dir": str(self.cache_dir)
            }

    def clear_cache(self, cache_type: str = "all"):
        if self.preprocessor:
            self.preprocessor.clear_cache(cache_type)
        else:
            import shutil
            if Path(self.cache_dir).exists():
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir(exist_ok=True)
                logger.info("Cleared optimized cache directory")


def create_optimized_glycan_dataloaders(data_path: str = "data/v12_glycan_binding.csv",
                                        embedder=None,
                                        test_size: float = 0.2,
                                        val_size: float = 0.1,
                                        batch_size: int = 32,
                                        max_pairs: Optional[int] = None,
                                        device: Optional[str] = None,
                                        cache_dir: str = "optimized_embeddings",
                                        use_precomputed: bool = True,
                                        use_clustering: bool = True,
                                        n_clusters: int = 10,
                                        sequence_identity: float = 0.5,
                                        fusion_method: str = "concat",
                                        protein_model: str = "650M",
                                        glycan_method: str = "lstm",
                                        glycan_vocab_path: Optional[str] = None,
                                        glycan_hidden_dims: Optional[List[int]] = None,
                                        glycan_readout: str = "mean",
                                        force_recompute: bool = False,
                                        save_splitter_path: Optional[str] = None,
                                        plot_analysis: bool = False,
                                        **kwargs) -> Dict[str, DataLoader]:

    logger.info("Creating OPTIMIZED glycan DataLoaders")

    clustering_params = {
        'n_clusters': n_clusters,
        'test_size': test_size,
        'val_size': val_size,
        'random_state': kwargs.get('random_state', 42),
        'sequence_identity': sequence_identity,
        'coverage': kwargs.get('coverage', 0.8),
        'use_pca': kwargs.get('use_pca', True)
    }

    loader = OptimizedEnhancedGlycanProteinDataLoader(
        data_path=data_path,
        embedder=embedder,
        device=device,
        cache_dir=cache_dir,
        use_precomputed=use_precomputed,
        use_clustering=use_clustering,
        clustering_params=clustering_params
    )

    return loader.create_train_val_test_loaders(
        batch_size=batch_size,
        max_pairs_per_split=max_pairs,
        fusion_method=fusion_method,
        protein_model=protein_model,
        glycan_method=glycan_method,
        glycan_vocab_path=glycan_vocab_path,
        glycan_hidden_dims=glycan_hidden_dims,
        glycan_readout=glycan_readout,
        force_recompute=force_recompute,
        save_splitter_path=save_splitter_path,
        plot_analysis=plot_analysis,
        num_workers=20,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing OPTIMIZED Glycan PyTorch DataLoader")
    print("=" * 70)

    try:
        dataloaders = create_optimized_glycan_dataloaders(
            data_path="data/v12_glycan_binding.csv",
            batch_size=16,
            max_pairs=200,
            use_precomputed=True,
            use_clustering=True,
            n_clusters=5,
            sequence_identity=0.5,
            protein_model="650M",
            glycan_method="lstm",
            glycan_vocab_path="GlycanEmbedder_Package/glycoword_vocab.pkl",
            save_splitter_path="optimized_splitter.pkl",
            plot_analysis=True
        )

        train_loader = dataloaders['train']

        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: embeddings {embeddings.shape}, targets {targets.shape}")
            print(f"Device: embeddings on {embeddings.device}, targets on {targets.device}")
            if batch_idx >= 2:
                break

        print("OPTIMIZED DataLoader test completed successfully!")

    except Exception as e:
        print(f"Error in optimized test: {e}")
        import traceback
        traceback.print_exc()
