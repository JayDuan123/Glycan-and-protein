import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinClusteringSplitter:
    def __init__(self,
                 n_clusters: int = 10,
                 test_size: float = 0.1,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 sequence_identity: float = 0.5,
                 coverage: float = 0.8,
                 mmseqs_binary: str = "mmseqs",
                 temp_dir: Optional[str] = None,
                 use_pca: bool = True,
                 pca_components: int = 50,
                 normalize_embeddings: bool = True,
                 **kwargs):
        self.n_clusters = n_clusters
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.sequence_identity = sequence_identity
        self.coverage = coverage
        self.mmseqs_binary = mmseqs_binary
        self.temp_dir = temp_dir or tempfile.gettempdir()
        ignored_params = []
        if not use_pca:
            ignored_params.append("use_pca=False")
        if pca_components != 50:
            ignored_params.append(f"pca_components={pca_components}")
        if not normalize_embeddings:
            ignored_params.append("normalize_embeddings=False")
        if kwargs:
            ignored_params.extend([f"{k}={v}" for k, v in kwargs.items()])
        if ignored_params:
            logger.info(f"MMseqs2 splitter ignoring legacy parameters: {', '.join(ignored_params)}")
        self.cluster_assignments = None
        self.split_assignments = None
        self.cluster_representatives = None
        self.cluster_stats = None
        np.random.seed(random_state)
        self._check_mmseqs2()

    def _check_mmseqs2(self):
        try:
            result = subprocess.run([self.mmseqs_binary, "version"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Found MMseqs2: {version}")
            else:
                logger.error("MMseqs2 found but version check failed")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"MMseqs2 not found or not working: {e}")
            raise RuntimeError("MMseqs2 is required but not available")

    def _write_fasta(self, protein_sequences: List[str], fasta_path: str):
        with open(fasta_path, 'w') as f:
            for i, seq in enumerate(protein_sequences):
                f.write(f">protein_{i}\n{seq}\n")
        logger.info(f"Wrote {len(protein_sequences)} sequences to {fasta_path}")

    def _run_mmseqs2_clustering(self, fasta_path: str, output_dir: str) -> str:
        logger.info(f"Running MMseqs2 clustering with identity={self.sequence_identity}, coverage={self.coverage}")
        db_path = os.path.join(output_dir, "sequenceDB")
        cmd = [self.mmseqs_binary, "createdb", fasta_path, db_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"MMseqs2 createdb failed: {result.stderr}")
                raise RuntimeError("MMseqs2 database creation failed")
        except subprocess.TimeoutExpired:
            logger.error("MMseqs2 createdb timed out")
            raise RuntimeError("MMseqs2 database creation timed out")

        cluster_db = os.path.join(output_dir, "clusterDB")
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        cmd = [
            self.mmseqs_binary, "cluster", db_path, cluster_db, tmp_dir,
            "--min-seq-id", str(self.sequence_identity),
            "-c", str(self.coverage),
            "--cov-mode", "1",
            "--cluster-mode", "0",
            "-s", "7.5",
            "--threads", "4"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"MMseqs2 cluster failed: {result.stderr}")
                raise RuntimeError("MMseqs2 clustering failed")
        except subprocess.TimeoutExpired:
            logger.error("MMseqs2 clustering timed out")
            raise RuntimeError("MMseqs2 clustering timed out")

        cluster_tsv = os.path.join(output_dir, "clusters.tsv")
        cmd = [self.mmseqs_binary, "createtsv", db_path, db_path, cluster_db, cluster_tsv]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"MMseqs2 createtsv failed: {result.stderr}")
                raise RuntimeError("MMseqs2 TSV creation failed")
        except subprocess.TimeoutExpired:
            logger.error("MMseqs2 createtsv timed out")
            raise RuntimeError("MMseqs2 clustering TSV creation timed out")

        logger.info("MMseqs2 clustering completed successfully")
        return cluster_tsv

    def _parse_clusters(self, cluster_tsv: str, protein_sequences: List[str]) -> Dict[str, int]:
        logger.info(f"Parsing clustering results from {cluster_tsv}")
        cluster_data = {}
        representative_to_cluster = {}
        cluster_id = 0

        with open(cluster_tsv, 'r') as f:
            for line in f:
                representative, member = line.strip().split('\t')
                rep_idx = int(representative.split('_')[1])
                mem_idx = int(member.split('_')[1])
                if representative not in representative_to_cluster:
                    representative_to_cluster[representative] = cluster_id
                    cluster_id += 1
                cluster_data[mem_idx] = representative_to_cluster[representative]

        self.cluster_assignments = {}
        for i, seq in enumerate(protein_sequences):
            self.cluster_assignments[seq] = cluster_data.get(i, cluster_id)
            if i not in cluster_data:
                cluster_id += 1

        cluster_counts = defaultdict(int)
        for cid in self.cluster_assignments.values():
            cluster_counts[cid] += 1

        n_clusters = len(cluster_counts)
        logger.info(f"MMseqs2 created {n_clusters} clusters")

        self.cluster_stats = {
            'n_clusters': n_clusters,
            'cluster_sizes': dict(cluster_counts),
            'largest_cluster': max(cluster_counts.values()),
            'smallest_cluster': min(cluster_counts.values()),
            'mean_cluster_size': np.mean(list(cluster_counts.values())),
            'median_cluster_size': np.median(list(cluster_counts.values()))
        }

        logger.info("Cluster distribution:")
        logger.info(f"  Total clusters: {n_clusters}")
        logger.info(f"  Largest cluster: {self.cluster_stats['largest_cluster']} proteins")
        logger.info(f"  Smallest cluster: {self.cluster_stats['smallest_cluster']} proteins")
        logger.info(f"  Mean cluster size: {self.cluster_stats['mean_cluster_size']:.1f}")
        logger.info(f"  Median cluster size: {self.cluster_stats['median_cluster_size']:.1f}")

        return self.cluster_assignments

    def fit_clusters(self,
                     protein_embeddings: np.ndarray,
                     protein_sequences: List[str]) -> Dict[str, int]:
        logger.info(f"Clustering {len(protein_sequences)} proteins using MMseqs2...")
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
            fasta_path = os.path.join(temp_dir, "sequences.fasta")
            self._write_fasta(protein_sequences, fasta_path)
            cluster_tsv = self._run_mmseqs2_clustering(fasta_path, temp_dir)
            cluster_assignments = self._parse_clusters(cluster_tsv, protein_sequences)
        logger.info("MMseqs2 clustering completed successfully")
        return cluster_assignments

    def assign_cluster_splits(self) -> Dict[int, str]:
        if self.cluster_assignments is None:
            raise ValueError("Must fit clusters first using fit_clusters()")
        cluster_ids = list(set(self.cluster_assignments.values()))
        cluster_ids.sort()
        n_clusters = len(cluster_ids)
        logger.info(f"Assigning {n_clusters} clusters to train/val/test splits")
        np.random.seed(self.random_state)
        np.random.shuffle(cluster_ids)

        n_test_clusters = max(1, int(n_clusters * self.test_size))
        n_val_clusters = max(1, int(n_clusters * self.val_size))
        n_train_clusters = n_clusters - n_test_clusters - n_val_clusters
        if n_train_clusters < 1:
            logger.warning("Too few clusters for requested split sizes, adjusting...")
            n_test_clusters = max(1, n_clusters // 3)
            n_val_clusters = max(1, n_clusters // 3)
            n_train_clusters = n_clusters - n_test_clusters - n_val_clusters

        test_clusters = cluster_ids[:n_test_clusters]
        val_clusters = cluster_ids[n_test_clusters:n_test_clusters + n_val_clusters]
        train_clusters = cluster_ids[n_test_clusters + n_val_clusters:]

        self.split_assignments = {}
        for cid in test_clusters:
            self.split_assignments[cid] = 'test'
        for cid in val_clusters:
            self.split_assignments[cid] = 'val'
        for cid in train_clusters:
            self.split_assignments[cid] = 'train'

        logger.info("Cluster-to-split assignments:")
        logger.info(f"  Train clusters: {len(train_clusters)} ({n_train_clusters})")
        logger.info(f"  Val clusters: {len(val_clusters)} ({n_val_clusters})")
        logger.info(f"  Test clusters: {len(test_clusters)} ({n_test_clusters})")

        return self.split_assignments

    def get_protein_splits(self) -> Dict[str, List[str]]:
        if self.cluster_assignments is None or self.split_assignments is None:
            raise ValueError("Must fit clusters and assign splits first")

        splits = {'train': [], 'val': [], 'test': []}
        for protein_seq, cluster_id in self.cluster_assignments.items():
            split_name = self.split_assignments[cluster_id]
            splits[split_name].append(protein_seq)

        logger.info("Final protein split distribution:")
        total_proteins = sum(len(proteins) for proteins in splits.values())
        for split_name, proteins in splits.items():
            n_clusters = len(set(self.cluster_assignments[p] for p in proteins))
            logger.info(f"  {split_name}: {len(proteins)} proteins "
                        f"({len(proteins) / total_proteins * 100:.1f}%) "
                        f"across {n_clusters} clusters")

        return splits

    def fit_and_split(self,
                      protein_embeddings: np.ndarray,
                      protein_sequences: List[str]) -> Dict[str, List[str]]:
        self.fit_clusters(protein_embeddings, protein_sequences)
        self.assign_cluster_splits()
        return self.get_protein_splits()

    def plot_clustering_analysis(self,
                                 protein_embeddings: np.ndarray,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 10)):
        if self.cluster_assignments is None:
            raise ValueError("Must fit clusters first")

        logger.info("Creating MMseqs2 clustering analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('MMseqs2 Protein Clustering Analysis', fontsize=16)

        cluster_sizes = list(self.cluster_stats['cluster_sizes'].values())
        axes[0, 0].hist(cluster_sizes, bins=min(20, len(set(cluster_sizes))), alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster Size (Number of Proteins)')
        axes[0, 0].set_ylabel('Number of Clusters')
        axes[0, 0].grid(True, alpha=0.3)

        splits = self.get_protein_splits()
        split_sizes = [len(proteins) for proteins in splits.values()]
        split_names = list(splits.keys())
        colors = ['blue', 'orange', 'red']
        bars = axes[0, 1].bar(split_names, split_sizes, color=colors, alpha=0.7)
        axes[0, 1].set_title('Protein Split Distribution')
        axes[0, 1].set_ylabel('Number of Proteins')
        for bar, size in zip(bars, split_sizes):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{size}', ha='center', va='bottom')

        cluster_counts_per_split = {}
        for split_name in splits.keys():
            proteins = splits[split_name]
            clusters = set(self.cluster_assignments[p] for p in proteins)
            cluster_counts_per_split[split_name] = len(clusters)

        bars = axes[1, 0].bar(cluster_counts_per_split.keys(),
                             cluster_counts_per_split.values(),
                             color=colors, alpha=0.7)
        axes[1, 0].set_title('Clusters per Split')
        axes[1, 0].set_ylabel('Number of Clusters')
        for bar, count in zip(bars, cluster_counts_per_split.values()):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{count}', ha='center', va='bottom')

        summary_text = f"""MMseqs2 Clustering Summary:

Identity Threshold: {self.sequence_identity:.1%}
Coverage Threshold: {self.coverage:.1%}

Results:
• Total Clusters: {self.cluster_stats['n_clusters']}
• Largest Cluster: {self.cluster_stats['largest_cluster']} proteins
• Smallest Cluster: {self.cluster_stats['smallest_cluster']} proteins
• Mean Size: {self.cluster_stats['mean_cluster_size']:.1f}
• Median Size: {self.cluster_stats['median_cluster_size']:.1f}

Split Information:
• Train: {len(splits['train'])} proteins
• Val: {len(splits['val'])} proteins  
• Test: {len(splits['test'])} proteins
• Train/Val/Test clusters: {cluster_counts_per_split['train']}/{cluster_counts_per_split['val']}/{cluster_counts_per_split['test']}"""

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Clustering Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved MMseqs2 clustering analysis plot to: {save_path}")
        plt.show()

    def save_splitter(self, save_path: str):
        save_data = {
            'n_clusters': self.n_clusters,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'sequence_identity': self.sequence_identity,
            'coverage': self.coverage,
            'mmseqs_binary': self.mmseqs_binary,
            'temp_dir': self.temp_dir,
            'cluster_assignments': self.cluster_assignments,
            'split_assignments': self.split_assignments,
            'cluster_stats': self.cluster_stats,
            'splitter_type': 'mmseqs2'
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved MMseqs2 splitter to: {save_path}")

    @classmethod
    def load_splitter(cls, load_path: str) -> 'ProteinClusteringSplitter':
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        if save_data.get('splitter_type') != 'mmseqs2':
            logger.warning("Loading splitter that may not be MMseqs2-based")

        splitter = cls(
            n_clusters=save_data['n_clusters'],
            test_size=save_data['test_size'],
            val_size=save_data['val_size'],
            random_state=save_data['random_state'],
            sequence_identity=save_data['sequence_identity'],
            coverage=save_data['coverage'],
            mmseqs_binary=save_data['mmseqs_binary'],
            temp_dir=save_data['temp_dir'],
            use_pca=save_data.get('use_pca', True),
            pca_components=save_data.get('pca_components', 50),
            normalize_embeddings=save_data.get('normalize_embeddings', True)
        )

        splitter.cluster_assignments = save_data['cluster_assignments']
        splitter.split_assignments = save_data['split_assignments']
        splitter.cluster_stats = save_data['cluster_stats']

        logger.info(f"Loaded MMseqs2 splitter from: {load_path}")
        return splitter

    def get_cluster_info(self) -> Dict:
        if not self.cluster_stats:
            return {"error": "No clustering performed yet"}
        info = self.cluster_stats.copy()
        if self.split_assignments:
            splits = self.get_protein_splits()
            info.update({
                'split_protein_counts': {name: len(proteins) for name, proteins in splits.items()},
                'split_cluster_counts': {
                    split_name: len(set(self.cluster_assignments[p] for p in proteins))
                    for split_name, proteins in splits.items()
                }
            })
        return info


def create_clustered_splits(protein_embeddings: np.ndarray,
                            protein_sequences: List[str],
                            n_clusters: int = 10,
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            random_state: int = 42,
                            sequence_identity: float = 0.5,
                            coverage: float = 0.8,
                            plot_analysis: bool = True,
                            save_splitter_path: Optional[str] = None,
                            mmseqs_binary: str = "mmseqs",
                            use_pca: bool = True,
                            pca_components: int = 50,
                            normalize_embeddings: bool = True,
                            **kwargs) -> Tuple[Dict[str, List[str]], 'ProteinClusteringSplitter']:
    ignored_params = []
    if not use_pca:
        ignored_params.append("use_pca=False")
    if pca_components != 50:
        ignored_params.append(f"pca_components={pca_components}")
    if not normalize_embeddings:
        ignored_params.append("normalize_embeddings=False")
    if kwargs:
        ignored_params.extend([f"{k}={v}" for k, v in kwargs.items()])
    if ignored_params:
        logger.info(f"MMseqs2 clustering ignoring legacy parameters: {', '.join(ignored_params)}")

    splitter = ProteinClusteringSplitter(
        n_clusters=n_clusters,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        sequence_identity=sequence_identity,
        coverage=coverage,
        mmseqs_binary=mmseqs_binary
    )

    protein_splits = splitter.fit_and_split(protein_embeddings, protein_sequences)

    if plot_analysis:
        try:
            splitter.plot_clustering_analysis(protein_embeddings)
        except ImportError:
            logger.warning("Matplotlib not available, skipping clustering plots")

    if save_splitter_path:
        splitter.save_splitter(save_splitter_path)

    return protein_splits, splitter


if __name__ == "__main__":
    print("Testing MMseqs2 protein clustering splitter...")
    protein_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGK",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPK"
    ] * 10
    mock_embeddings = np.random.randn(len(protein_sequences), 100)

    try:
        protein_splits, splitter = create_clustered_splits(
            protein_embeddings=mock_embeddings,
            protein_sequences=protein_sequences,
            n_clusters=5,
            plot_analysis=True,
            save_splitter_path="test_mmseqs2_splitter.pkl"
        )

        print("MMseqs2 clustering test completed successfully!")
        print(f"Split sizes: train={len(protein_splits['train'])}, "
              f"val={len(protein_splits['val'])}, test={len(protein_splits['test'])}")

        info = splitter.get_cluster_info()
        print(f"Cluster info: {info}")

    except Exception as e:
        print(f"Error in MMseqs2 clustering test: {e}")
        import traceback
        traceback.print_exc()
