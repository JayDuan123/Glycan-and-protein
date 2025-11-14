"""
Enhanced Binding Strength Predictor
Integrates precomputed embeddings, clustering-based splits, and improved training
Maintains API compatibility while adding advanced features
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr  # ADDED: For Spearman's rho calculation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import enhanced modules
from dataloader.glycan_dataloader_cpu_v2 import EnhancedGlycanProteinDataLoader, create_enhanced_glycan_dataloaders
from network.binding_strength_networks import BindingStrengthNetworkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBindingStrengthPredictor:
    """
    Enhanced binding strength predictor with precomputed embeddings and clustering
    Maintains backward compatibility with original BindingStrengthPredictor API
    """

    def __init__(self,
                 # Original embedder parameters (for backward compatibility)
                 protein_model: str = "650M",
                 protein_model_dir: str = "resources/esm-model-weights",
                 glycan_method: str = "lstm",
                 glycan_vocab_path: Optional[str] = None,
                 glycan_hidden_dims: Optional[List[int]] = None,
                 glycan_readout: str = "mean",
                 fusion_method: str = "concat",

                 # Network parameters
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,

                 # Enhanced features
                 use_precomputed: bool = True,
                 use_clustering: bool = True,
                 cache_dir: str = "enhanced_embeddings",
                 clustering_params: Optional[Dict] = None,

                 # Training parameters
                 device: Optional[str] = None,
                 random_seed: int = 42):
        """
        Initialize the Enhanced Binding Strength Predictor

        Args:
            protein_model: ESM2 model size ("650M" or "3B")
            protein_model_dir: Directory for protein model weights
            glycan_method: Glycan embedding method
            glycan_vocab_path: Path to glycan vocabulary file
            glycan_hidden_dims: Hidden dimensions for glycan embedder
            glycan_readout: Readout function for graph-based methods
            fusion_method: "concat" or "attention"
            network_type: Type of neural network
            network_config: Configuration for the neural network
            use_precomputed: Whether to use precomputed embeddings
            use_clustering: Whether to use clustering-based splits
            cache_dir: Directory for embedding cache
            clustering_params: Parameters for clustering
            device: Device to use (None = auto-detect)
            random_seed: Random seed for reproducibility
        """
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Store parameters
        self.protein_model = protein_model
        self.protein_model_dir = protein_model_dir
        self.glycan_method = glycan_method
        self.glycan_vocab_path = glycan_vocab_path
        self.glycan_hidden_dims = glycan_hidden_dims
        self.glycan_readout = glycan_readout
        self.fusion_method = fusion_method

        # Enhanced features
        self.use_precomputed = use_precomputed
        self.use_clustering = use_clustering
        self.cache_dir = cache_dir
        self.clustering_params = clustering_params or {
            'n_clusters': 10,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': random_seed,
            'use_pca': True
        }

        # Initialize network configuration
        self.network_type = network_type
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)

        # Components (initialized during training)
        self.data_loader = None
        self.preprocessor = None
        self.splitter = None
        self.model = None
        self.embedding_dim = None

        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'setup_time': 0,
            'training_time': 0
        }

        logger.info(f"Enhanced predictor initialized")
        logger.info(f"Features: precomputed={use_precomputed}, clustering={use_clustering}")

    def setup_preprocessing(self,
                            data_path: str,
                            force_recompute: bool = False,
                            **kwargs):
        """
        Setup preprocessing components (embeddings and clustering)

        Args:
            data_path: Path to dataset
            force_recompute: Whether to recompute existing embeddings
            **kwargs: Additional arguments for preprocessing
        """
        if not self.use_precomputed:
            logger.info("Precomputed embeddings disabled, skipping preprocessing setup")
            return

        logger.info("Setting up enhanced preprocessing...")
        start_time = time.time()

        # Initialize data loader
        self.data_loader = EnhancedGlycanProteinDataLoader(
            data_path=data_path,
            device=self.device,
            cache_dir=self.cache_dir,
            use_precomputed=self.use_precomputed,
            use_clustering=self.use_clustering,
            clustering_params=self.clustering_params
        )

        # Setup precomputed embeddings
        self.data_loader.setup_precomputed_embeddings(
            protein_model=self.protein_model,
            glycan_method=self.glycan_method,
            glycan_vocab_path=self.glycan_vocab_path,
            glycan_hidden_dims=self.glycan_hidden_dims,
            glycan_readout=self.glycan_readout,
            force_recompute=force_recompute,
            **kwargs
        )

        # Setup clustering if enabled
        if self.use_clustering:
            self.data_loader.setup_clustering_splits()

        setup_time = time.time() - start_time
        self.training_history['setup_time'] = setup_time
        logger.info(f"Preprocessing setup completed in {setup_time:.1f}s")

    def prepare_data(self,
                     data_path: str,
                     batch_size: int = 32,
                     normalize_targets: bool = True,
                     max_pairs_per_split: Optional[int] = None,
                     force_recompute: bool = False,
                     **kwargs) -> Dict[str, DataLoader]:
        """
        Prepare data for training and evaluation with enhanced features

        Args:
            data_path: Path to dataset file
            batch_size: Batch size for data loaders
            normalize_targets: Whether to normalize target values
            max_pairs_per_split: Maximum pairs per split
            force_recompute: Whether to recompute embeddings
            **kwargs: Additional arguments

        Returns:
            Dictionary containing train, validation, and test data loaders
        """
        logger.info(f"Preparing enhanced data from {data_path}...")

        # Setup preprocessing
        self.setup_preprocessing(data_path, force_recompute=force_recompute, **kwargs)

        # Create enhanced dataloaders
        if self.data_loader is None:
            # Fallback to direct creation
            dataloaders = create_enhanced_glycan_dataloaders(
                data_path=data_path,
                batch_size=batch_size,
                max_pairs=max_pairs_per_split,
                device=self.device,
                cache_dir=self.cache_dir,
                use_precomputed=self.use_precomputed,
                use_clustering=self.use_clustering,
                protein_model=self.protein_model,
                glycan_method=self.glycan_method,
                glycan_vocab_path=self.glycan_vocab_path,
                **self.clustering_params,
                **kwargs
            )
        else:
            # Use existing data loader
            dataloaders = self.data_loader.create_train_val_test_loaders(
                batch_size=batch_size,
                normalize_targets=normalize_targets,
                max_pairs_per_split=max_pairs_per_split,
                fusion_method=self.fusion_method,
                **kwargs
            )

        # Initialize model based on data
        if 'train' in dataloaders:
            sample_batch = next(iter(dataloaders['train']))
            self.embedding_dim = sample_batch[0].shape[1]

            self.model = BindingStrengthNetworkFactory.create_network(
                self.network_type, self.embedding_dim, **self.network_config
            ).to(self.device)

            logger.info(f"Model initialized: embedding_dim={self.embedding_dim}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Store scaler reference
        if 'target_scaler' in dataloaders:
            self.scaler = dataloaders['target_scaler']

        return dataloaders

    def train(self,
              dataloaders: Dict[str, DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10,
              min_delta: float = 1e-4,
              scheduler_config: Optional[Dict] = None) -> Dict:
        """
        Train the model with enhanced features

        Args:
            dataloaders: Dictionary with train/val/test data loaders
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for L2 regularization
            patience: Early stopping patience
            min_delta: Minimum change for early stopping
            scheduler_config: Configuration for learning rate scheduler

        Returns:
            Training history
        """
        logger.info(f"Starting enhanced training for {num_epochs} epochs...")
        training_start_time = time.time()

        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data() first.")

        # Initialize training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize scheduler if config provided
        if scheduler_config:
            scheduler_type = scheduler_config.pop('type', 'reduce_on_plateau')
            if scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **scheduler_config
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=num_epochs, **scheduler_config
                )

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        train_loader = dataloaders['train']
        val_loader = dataloaders.get('val', None)

        logger.info(f"Training batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)

            # Validation phase
            if val_loader:
                val_loss, val_metrics = self._eval_epoch(val_loader)
            else:
                val_loss, val_metrics = train_loss, train_metrics

            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # MODIFIED: Log progress every epoch instead of every 10 epochs
            logger.info(
                f"Epoch {epoch:3d}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Train R²={train_metrics['r2']:.4f}, Val R²={val_metrics['r2']:.4f}, "
                f"Train Spearman={train_metrics['spearman']:.4f}, Val Spearman={val_metrics['spearman']:.4f}"
            )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        self.is_fitted = True
        training_time = time.time() - training_start_time
        self.training_history['training_time'] = training_time

        logger.info(f"Training completed in {training_time:.1f}s")
        return self.training_history

    def _train_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in tqdm(data_loader, desc="Training", leave=False):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def _eval_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(data_loader, desc="Validation", leave=False):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def _get_actual_glycan_params(self) -> Dict:
        """Extract actual glycan parameters used"""
        if self.data_loader and hasattr(self.data_loader, 'preprocessor'):
            preprocessor = self.data_loader.preprocessor
            if preprocessor and hasattr(preprocessor, 'glycan_params'):
                return preprocessor.glycan_params
        
        # Fallback to stored parameters
        return {
            'method': self.glycan_method,
            'hidden_dims': self.glycan_hidden_dims,
            'readout': self.glycan_readout,
            'vocab_path': self.glycan_vocab_path
        }

    def _calculate_metrics(self, targets: List[float], predictions: List[float]) -> Dict:
        """Calculate evaluation metrics"""
        targets = np.array(targets)
        predictions = np.array(predictions)

        # ADDED: Calculate Spearman's rho
        spearman_corr, spearman_p = spearmanr(targets, predictions)

        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions),
            'spearman': spearman_corr  # ADDED: Spearman's rank correlation
        }

    def predict(self,
                pairs: List[Tuple[str, str]],
                batch_size: int = 32,
                return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict binding strengths for new pairs
        Enhanced version that can use precomputed embeddings if available

        Args:
            pairs: List of (glycan_iupac, protein_sequence) tuples
            batch_size: Batch size for processing
            return_numpy: Whether to return numpy array

        Returns:
            Predicted binding strengths
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        logger.info(f"Predicting binding strengths for {len(pairs)} pairs...")

        if self.use_precomputed and self.data_loader:
            # Use enhanced prediction with precomputed embeddings
            return self._predict_with_precomputed(pairs, batch_size, return_numpy)
        else:
            # Fallback to original prediction method
            return self._predict_original(pairs, batch_size, return_numpy)

    def _predict_with_precomputed(self, pairs, batch_size, return_numpy):
        """Predict using precomputed embeddings"""
        # This would require creating a temporary dataset
        # For now, fallback to original method
        return self._predict_original(pairs, batch_size, return_numpy)

    def _predict_original(self, pairs, batch_size, return_numpy):
        """Original prediction method using embedder"""
        from embedder.Integrated_Embedder import GlycanProteinPairEmbedder

        # Initialize embedder if needed
        embedder = GlycanProteinPairEmbedder(
            protein_model=self.protein_model,
            protein_model_dir=self.protein_model_dir,
            glycan_method=self.glycan_method,
            glycan_vocab_path=self.glycan_vocab_path,
            glycan_hidden_dims=self.glycan_hidden_dims,
            glycan_readout=self.glycan_readout,
            fusion_method=self.fusion_method,
            device=self.device
        )

        # Generate embeddings
        embeddings = embedder.embed_pairs(pairs, batch_size=batch_size, return_numpy=False)

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size].to(self.device)
                pred = self.model(batch).squeeze()
                predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)

        # Denormalize if scaler was used
        if hasattr(self.scaler, 'scale_'):
            predictions_np = predictions.cpu().numpy().reshape(-1, 1)
            predictions_np = self.scaler.inverse_transform(predictions_np).flatten()
            predictions = torch.FloatTensor(predictions_np)

        if return_numpy:
            return predictions.cpu().numpy()
        else:
            return predictions

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a dataset"""
        _, metrics = self._eval_epoch(data_loader)
        return metrics

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot enhanced training history with additional metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Training Results', fontsize=16)

        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # R² curves
        train_r2 = [m['r2'] for m in self.training_history['train_metrics']]
        val_r2 = [m['r2'] for m in self.training_history['val_metrics']]
        axes[0, 1].plot(train_r2, label='Train R²')
        axes[0, 1].plot(val_r2, label='Val R²')
        axes[0, 1].set_title('R² Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # ADDED: Spearman's rho curves
        train_spearman = [m['spearman'] for m in self.training_history['train_metrics']]
        val_spearman = [m['spearman'] for m in self.training_history['val_metrics']]
        axes[0, 2].plot(train_spearman, label='Train Spearman')
        axes[0, 2].plot(val_spearman, label='Val Spearman')
        axes[0, 2].set_title('Spearman Correlation Curves')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Spearman ρ')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # MAE curves
        train_mae = [m['mae'] for m in self.training_history['train_metrics']]
        val_mae = [m['mae'] for m in self.training_history['val_metrics']]
        axes[1, 0].plot(train_mae, label='Train MAE')
        axes[1, 0].plot(val_mae, label='Val MAE')
        axes[1, 0].set_title('MAE Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Performance summary
        final_metrics = {
            'Setup Time': f"{self.training_history.get('setup_time', 0):.1f}s",
            'Training Time': f"{self.training_history.get('training_time', 0):.1f}s",
            'Best Val R²': f"{max(val_r2):.4f}" if val_r2 else "N/A",
            'Final Val R²': f"{val_r2[-1]:.4f}" if val_r2 else "N/A",
            'Best Val Spearman': f"{max(val_spearman):.4f}" if val_spearman else "N/A",
            'Final Val Spearman': f"{val_spearman[-1]:.4f}" if val_spearman else "N/A",
        }

        # Convert metrics to text
        metrics_text = "\n".join([f"{k}: {v}" for k, v in final_metrics.items()])
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')

        # Detailed parameter configuration
        protein_dim_map = {"650M": 1280, "3B": 2560}
        protein_dim = protein_dim_map.get(self.protein_model, 1280)
        
        # Get network parameters
        network_params = self.network_config
        hidden_dims_str = str(network_params.get('hidden_dims', ['N/A']))
        
        # Get training parameters (passed from training function)
        train_params = training_params or {}
        
        features_text = f"""Training Configuration:
            - Batch Size: {train_params.get('batch_size', 'N/A')}
            - Learning Rate: {train_params.get('learning_rate', 'N/A')}
            - Weight Decay: {train_params.get('weight_decay', 'N/A')}
            - Patience: {train_params.get('patience', 'N/A')}
            - Device: {self.device}
            
            Protein Embedder:
            - Model: ESM-2b-{self.protein_model}
            - Embedding Dim: {protein_dim}
            - Model Dir: {self.protein_model_dir.split('/')[-1] if self.protein_model_dir else 'default'}
            
            Glycan Embedder:
            - Method: {self.glycan_method.upper()}
            - Hidden Dims: {self.glycan_hidden_dims or 'default'}
            - Readout: {self.glycan_readout}
            - Vocab: {'✓' if self.glycan_vocab_path else '✗'}
            
            Combined Embedding:
            - Fusion: {self.fusion_method}
            - Input Dim: {self.embedding_dim or 'N/A'}
            
            Prediction Model:
            - Type: {self.network_type.upper()}
            - Hidden Dims: {hidden_dims_str}
            - Activation: {network_params.get('activation', 'N/A')}
            - Dropout: {network_params.get('dropout', 'N/A')}
            - Batch Norm: {'✓' if network_params.get('batch_norm') else '✗'}
            
            Enhanced Features:
            - Precomputed: {self.use_precomputed}
            - Clustering: {self.use_clustering}
            - N Clusters: {self.clustering_params.get('n_clusters', 'N/A')}
            - Cache Dir: {self.cache_dir.split('/')[-1]}
        """

        axes[1, 2].text(0.05, 0.95, features_text, transform=axes[1, 2].transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Detailed Parameters')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved enhanced training history plot to {save_path}")

        plt.show()

    def train(self, dataloaders, num_epochs=100, learning_rate=1e-3, 
              weight_decay=1e-4, patience=10, min_delta=1e-4, 
              scheduler_config=None, **kwargs):
        """Train the model and store parameters for visualization"""
        
        # ... existing training code ...
        
        # Store training parameters for visualization
        self._last_training_params = {
            'batch_size': dataloaders['train'].batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'patience': patience,
            'num_epochs': num_epochs,
            'scheduler_config': scheduler_config
        }
        
        # ... rest of training code ...
        
        return self.training_history

    # Update plot_training_history calls to include parameters
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot enhanced training history with stored parameters"""
        training_params = getattr(self, '_last_training_params', {})
        self._plot_training_history_with_params(save_path, training_params)

        # Convert metrics to text
        metrics_text = "\n".join([f"{k}: {v}" for k, v in final_metrics.items()])
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')

        # Enhancement features
        features_text = f"""Enhanced Features:

            ✓ Precomputed Embeddings
            ✓ Clustering-based Splits  
            ✓ Memory Optimization
            ✓ Faster Data Loading
            ✓ Better Generalization
            ✓ Spearman's Rho Metric
            
            Network: {self.network_type}
            Fusion: {self.fusion_method}
            Device: {self.device}
        """

        axes[1, 2].text(0.1, 0.5, features_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Enhancement Features')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved enhanced training history plot to {save_path}")

        plt.show()

    def save_model(self, path: str):
        """Save enhanced model and configuration"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'network_type': self.network_type,
            'network_config': self.network_config,
            'embedding_dim': self.embedding_dim,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted,

            # Enhanced features
            'use_precomputed': self.use_precomputed,
            'use_clustering': self.use_clustering,
            'cache_dir': self.cache_dir,
            'clustering_params': self.clustering_params,

            # Original parameters
            'protein_model': self.protein_model,
            'glycan_method': self.glycan_method,
            'fusion_method': self.fusion_method
        }

        torch.save(save_dict, path)
        logger.info(f"Saved enhanced model to {path}")

    def load_model(self, path: str):
        """Load enhanced model and configuration"""
        save_dict = torch.load(path, map_location=self.device)

        # Load original parameters
        self.network_type = save_dict['network_type']
        self.network_config = save_dict['network_config']
        self.embedding_dim = save_dict['embedding_dim']

        # Load enhanced parameters if available
        self.use_precomputed = save_dict.get('use_precomputed', False)
        self.use_clustering = save_dict.get('use_clustering', False)
        self.cache_dir = save_dict.get('cache_dir', 'enhanced_embeddings')
        self.clustering_params = save_dict.get('clustering_params', {})

        # Recreate model
        self.model = BindingStrengthNetworkFactory.create_network(
            self.network_type, self.embedding_dim, **self.network_config
        ).to(self.device)

        # Load state
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.scaler = save_dict['scaler']
        self.training_history = save_dict['training_history']
        self.is_fitted = save_dict['is_fitted']

        logger.info(f"Loaded enhanced model from {path}")

    def get_cache_info(self) -> Dict:
        """Get cache information"""
        if self.data_loader:
            return self.data_loader.get_cache_info()
        else:
            return {"message": "No data loader initialized"}


# Backward compatible convenience function
def load_data_from_file(file_path: str) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    Load glycan-protein pairs and binding strengths from file
    Enhanced version that can work with new data formats
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        pairs = [(item['glycan_iupac'], item['protein_sequence']) for item in data]
        strengths = [item['binding_strength'] for item in data]

    elif file_path.suffix.lower() in ['.csv', '.tsv']:
        separator = ',' if file_path.suffix.lower() == '.csv' else '\t'
        df = pd.read_csv(file_path, separator=separator)
        pairs = list(zip(df['glycan_iupac'], df['protein_sequence']))
        strengths = df['binding_strength'].tolist()

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(pairs)} pairs from {file_path}")
    return pairs, strengths


if __name__ == "__main__":
    # Example usage comparing original vs enhanced predictor
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    data_path = "../data/v12_glycan_binding.csv"

    print("Enhanced Binding Strength Predictor Example")
    print("=" * 50)

    try:
        # Initialize enhanced predictor
        predictor = EnhancedBindingStrengthPredictor(
            protein_model="650M",
            protein_model_dir="../resources/esm-model-weights",
            glycan_method="lstm",
            glycan_vocab_path=vocab_path,
            fusion_method="concat",
            network_type="mlp",
            network_config={
                "hidden_dims": [1024, 512, 256, 128],
                "dropout": 0.3,
                "activation": "relu"
            },
            use_precomputed=True,
            use_clustering=True,
            clustering_params={'n_clusters': 5}  # Small for testing
        )

        # Prepare data with enhanced features
        dataloaders = predictor.prepare_data(
            data_path=data_path,
            batch_size=16,
            max_pairs_per_split=200  # Small for testing
        )

        # Train model
        history = predictor.train(
        dataloaders,
        num_epochs=20,
        learning_rate=1e-3,
        patience=5,
        scheduler_config={'type': 'reduce_on_plateau', 'patience': 3}
        )
    
        # Plot with detailed parameters
        predictor.plot_training_history("enhanced_training_results.png")

        # Evaluate on test set
        test_metrics = predictor.evaluate(dataloaders['test'])
        print(f"\nEnhanced Test Results:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  Spearman's ρ: {test_metrics['spearman']:.4f}")  # ADDED

        # Show timing information
        print(f"\nPerformance:")
        print(f"  Setup time: {history.get('setup_time', 0):.1f}s")
        print(f"  Training time: {history.get('training_time', 0):.1f}s")

        # Plot results
        predictor.plot_training_history("enhanced_training_results.png")

        # Show cache info
        cache_info = predictor.get_cache_info()
        print(f"\nCache info: {cache_info}")

        # Save model
        predictor.save_model("enhanced_binding_model.pth")

        print("\nEnhanced predictor example completed successfully!")

    except Exception as e:
        logger.error(f"Error in enhanced predictor example: {e}")
        import traceback

        traceback.print_exc()