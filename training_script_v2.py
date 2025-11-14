import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
import logging
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from datetime import datetime

from optimized_dataloader_workflow import OptimizedEnhancedGlycanProteinDataLoader, create_optimized_glycan_dataloaders
from Integrated_Embedder import GlycanProteinPairEmbedder
from binding_strength_networks import BindingStrengthNetworkFactory
from embedding_preprocessor import EmbeddingPreprocessor
from clustering_splitter import ProteinClusteringSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPyTorchBindingPredictor:

    def __init__(self,
                 embedder: Optional[GlycanProteinPairEmbedder] = None,
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,
                 device: Optional[str] = None,
                 use_precomputed: bool = True,
                 use_clustering: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.use_precomputed = use_precomputed
        self.use_clustering = use_clustering
        if embedder is not None:
            self.embedding_dim = embedder.get_output_dim()
        else:
            self.embedding_dim = None
        self.network_type = network_type
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.target_scaler = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'train_spearman': [],
            'val_spearman': []
        }
        logger.info(f"Initialized enhanced predictor on {self.device}")
        logger.info(f"Use precomputed: {use_precomputed}, Use clustering: {use_clustering}")

    def _initialize_model(self, embedding_dim: int):
        if self.model is None:
            self.embedding_dim = embedding_dim
            self.model = BindingStrengthNetworkFactory.create_network(
                self.network_type, self.embedding_dim, **self.network_config
            ).to(self.device)
            logger.info(f"Initialized {self.network_type} network")
            logger.info(f"Embedding dim: {self.embedding_dim}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self,
              dataloaders: Dict[str, torch.utils.data.DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10) -> Dict:
        logger.info(f"Starting enhanced training for {num_epochs} epochs")
        self.target_scaler = dataloaders.get('target_scaler')
        train_loader = dataloaders['train']
        sample_batch = next(iter(train_loader))
        embedding_dim = sample_batch[0].shape[1]
        self._initialize_model(embedding_dim)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience // 2, factor=0.2, verbose=True
        )
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        val_loader = dataloaders.get('val', None)
        logger.info(f"Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")
        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss, train_r2, train_spearman = self._train_epoch(train_loader)
            if val_loader:
                val_loss, val_r2, val_spearman = self._eval_epoch(val_loader)
            else:
                val_loss, val_r2, val_spearman = train_loss, train_r2, train_spearman
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['train_spearman'].append(train_spearman)
            self.history['val_spearman'].append(val_spearman)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            logger.info(
                f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, "
                f"Train Spearman={train_spearman:.4f}, Val Spearman={val_spearman:.4f}, Time={epoch_time:.1f}s"
            )
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")
        return self.history

    def _train_epoch(self, dataloader) -> tuple:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, (embeddings, targets) in enumerate(pbar):
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(embeddings).squeeze(-1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)
        spearman = self._calculate_spearman(all_targets, all_preds)
        return avg_loss, r2, spearman

    def _eval_epoch(self, dataloader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Validation", leave=False):
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(embeddings).squeeze(-1)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)
        spearman = self._calculate_spearman(all_targets, all_preds)
        return avg_loss, r2, spearman

    def _calculate_r2(self, targets, predictions):
        from sklearn.metrics import r2_score
        return r2_score(targets, predictions)

    def _calculate_spearman(self, targets, predictions):
        spearman_corr, _ = spearmanr(targets, predictions)
        return spearman_corr

    def evaluate(self, dataloader) -> Dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        if self.model is None:
            raise ValueError("Model not initialized. Train the model first.")
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Evaluating", leave=False):
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(embeddings).squeeze(-1)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        if self.target_scaler:
            all_preds = self.target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
            all_targets = self.target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
        spearman_corr, _ = spearmanr(all_targets, all_preds)
        metrics = {
            'mse': mean_squared_error(all_targets, all_preds),
            'mae': mean_absolute_error(all_targets, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'r2': r2_score(all_targets, all_preds),
            'spearman': spearman_corr
        }
        return metrics, all_preds, all_targets


def run_enhanced_pytorch_pipeline():
    print("Running Enhanced PyTorch Glycan-Protein Binding Pipeline")
    print("=" * 80)
    data_path = "data/v12_glycan_binding.csv"
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    cache_dir = "enhanced_embedding_cache"
    use_precomputed = True
    use_clustering = True
    n_clusters = 10
    fusion_method = "concat"
    force_recompute = False
    num_epochs = 100
    learning_rate = 5e-5
    weight_decay = 0
    patience = 16
    protein_model = "650M"
    glycan_method = "lstm"
    network_type = "mlp"
    hidden_dims = [1280, 1280]
    dropout = 0
    activation = "gelu"

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")

    try:
        logger.info("Creating enhanced DataLoaders...")
        start_time = time.time()
        dataloaders = create_optimized_glycan_dataloaders(
            data_path=data_path,
            test_size=0.1,
            val_size=0.1,
            use_precomputed=use_precomputed,
            use_clustering=use_clustering,
            n_clusters=n_clusters,
            fusion_method=fusion_method,
            protein_model=protein_model,
            glycan_method=glycan_method,
            glycan_vocab_path=vocab_path,
            batch_size=batch_size,
            cache_dir=cache_dir,
            device=device,
            force_recompute=force_recompute
        )
        setup_time = time.time() - start_time
        logger.info(f"Enhanced DataLoader setup completed in {setup_time:.1f}s")
        temp_loader = OptimizedEnhancedGlycanProteinDataLoader(
            data_path=data_path,
            cache_dir=cache_dir,
            use_precomputed=use_precomputed,
            use_clustering=use_clustering
        )
        cache_info = temp_loader.get_cache_info()
        logger.info(f"Cache: {cache_info.get('protein_embeddings', 0)} proteins, "
                   f"{cache_info.get('glycan_embeddings', 0)} glycans, "
                   f"{cache_info.get('total_cache_size_mb', 0):.1f} MB total")
        sample_batch = next(iter(dataloaders['train']))
        actual_combined_dim = sample_batch[0].shape[1]
        protein_dim = 1280 if protein_model == "650M" else 2560
        glycan_dim = actual_combined_dim - protein_dim

        predictor = EnhancedPyTorchBindingPredictor(
            network_type=network_type,
            network_config={
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "activation": activation,
                "batch_norm": True
            },
            device=device,
            use_precomputed=use_precomputed,
            use_clustering=use_clustering
        )

        logger.info("Training enhanced model...")
        training_start = time.time()
        history = predictor.train(
            dataloaders=dataloaders,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience
        )
        training_time = time.time() - training_start

        logger.info("Evaluating enhanced model...")
        test_metrics, test_preds, test_targets = predictor.evaluate(dataloaders['test'])

        print(f"\nEnhanced Training Results:")
        print(f"Setup time: {setup_time:.1f}s")
        print(f"Training time: {training_time:.1f}s")
        print(f"Total time: {setup_time + training_time:.1f}s")
        print(f"\nModel Performance:")
        print(f"Best validation R²: {max(history['val_r2']):.4f}")
        print(f"Final validation R²: {history['val_r2'][-1]:.4f}")
        print(f"Best validation Spearman: {max(history['val_spearman']):.4f}")
        print(f"Final validation Spearman: {history['val_spearman'][-1]:.4f}")
        print(f"\nTest Results:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  Spearman's ρ: {test_metrics['spearman']:.4f}")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Enhanced Training Results with Clustering', fontsize=16)

            axes[0, 0].plot(history['train_loss'])
            axes[0, 0].plot(history['val_loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(history['train_r2'])
            axes[0, 1].plot(history['val_r2'])
            axes[0, 1].set_title('R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].grid(True, alpha=0.3)

            axes[0, 2].plot(history['train_spearman'])
            axes[0, 2].plot(history['val_spearman'])
            axes[0, 2].set_title('Spearman Correlation')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Spearman ρ')
            axes[0, 2].grid(True, alpha=0.3)

            axes[1, 0].scatter(test_targets, test_preds, alpha=0.6, s=20)
            min_val = min(test_targets.min(), test_preds.min())
            max_val = max(test_targets.max(), test_preds.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[1, 0].set_xlabel('Actual Binding Strength')
            axes[1, 0].set_ylabel('Predicted Binding Strength')
            axes[1, 0].grid(True, alpha=0.3)

            residuals = test_preds - test_targets
            axes[1, 1].scatter(test_targets, residuals, alpha=0.6, s=20)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Actual Binding Strength')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].grid(True, alpha=0.3)

            features_text = f"""
                Training Parameters:
                - Batch Size: {batch_size}
                - Learning Rate: {learning_rate}
                - Weight Decay: {weight_decay}
                - Epochs: {num_epochs}
                - Device: {device}
                
                Combined Embedding:
                - Total Dim: {actual_combined_dim}
                
                Prediction Model:
                - Type: {network_type.upper()}
                - Hidden Dims: {hidden_dims}
                - Activation: {activation}
                - Dropout: {dropout}
                - Batch Norm: ✓
                
                Enhanced Features:
                - Precomputed Cache: {use_precomputed}
                - Clustering Splits: {use_clustering}
                - Cache Size: {cache_info.get('total_cache_size_mb', 0):.1f} MB
                
                Performance:
                - Setup Time: {setup_time:.1f}s
                - Training Time: {training_time:.1f}s
                - Test R²: {test_metrics['r2']:.4f}
                - Test Spearman ρ: {test_metrics['spearman']:.4f}
            """
            axes[1, 2].text(0.05, 0.95, features_text, transform=axes[1, 2].transAxes,
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(f"results/enhanced_pytorch_pipeline_results_{datetime.now()}.png", dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            logger.info("Matplotlib not available, skipping plots")

        print(f"\nEnhanced PyTorch pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in enhanced pipeline: {e}")
        import traceback
        traceback.print_exc()


def compare_original_vs_enhanced():
    print("Comparing Original vs Enhanced Pipeline")
    print("=" * 50)
    print("This would compare:")
    print("1. Loading time")
    print("2. Generalization")
    print("3. Memory usage")
    print("4. Training stability")


def precompute_embeddings_only():
    print("Precomputing Embeddings Only")
    print("=" * 30)
    from embedding_preprocessor import preprocess_embeddings
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    data_path = "data/v12_glycan_binding.csv"
    preprocessor = preprocess_embeddings(
        data_path=data_path,
        cache_dir="enhanced_embedding_cache",
        protein_model="650M",
        glycan_method="lstm",
        glycan_vocab_path=vocab_path,
        batch_size=16,
        force_recompute=False
    )
    cache_info = preprocessor.get_cache_info()
    print(f"Precomputed embeddings:")
    print(f"  Proteins: {cache_info['protein_embeddings']}")
    print(f"  Glycans: {cache_info['glycan_embeddings']}")
    print(f"  Total size: {cache_info['total_cache_size_mb']:.1f} MB")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "precompute":
            precompute_embeddings_only()
        elif sys.argv[1] == "compare":
            compare_original_vs_enhanced()
        else:
            print("Available commands: precompute, compare")
    else:
        run_enhanced_pytorch_pipeline()
