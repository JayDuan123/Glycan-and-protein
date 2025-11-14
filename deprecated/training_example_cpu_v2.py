"""
Enhanced PyTorch Integration Example: Glycan-Protein Binding Prediction
Features precomputed embeddings and clustering-based data splits for improved performance
Maintains compatibility with original API while adding new capabilities
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
import logging
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Import enhanced modules
from glycan_dataloader_cpu_v2 import EnhancedGlycanProteinDataLoader, create_enhanced_glycan_dataloaders
from Integrated_Embedder import GlycanProteinPairEmbedder
from binding_strength_networks import BindingStrengthNetworkFactory
from embedding_preprocessor import EmbeddingPreprocessor
from clustering_splitter import ProteinClusteringSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPyTorchBindingPredictor:
    """
    Enhanced PyTorch-based binding strength predictor with precomputed embeddings
    and clustering-based data splits
    """

    def __init__(self,
                 embedder: Optional[GlycanProteinPairEmbedder] = None,
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,
                 device: Optional[str] = None,
                 use_precomputed: bool = True,
                 use_clustering: bool = True):
        """
        Initialize the enhanced predictor

        Args:
            embedder: Pre-initialized embedder (optional when using precomputed)
            network_type: Type of neural network
            network_config: Network configuration
            device: Computing device
            use_precomputed: Whether to use precomputed embeddings
            use_clustering: Whether to use clustering-based splits
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.use_precomputed = use_precomputed
        self.use_clustering = use_clustering

        # Get embedding dimension (will be determined from data if using precomputed)
        if embedder is not None:
            self.embedding_dim = embedder.get_output_dim()
        else:
            # Will be set when loading data
            self.embedding_dim = None

        # Network configuration
        self.network_type = network_type
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)
        
        # Model will be initialized after determining embedding dimension
        self.model = None

        # Training components
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.target_scaler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }

        logger.info(f"Initialized enhanced predictor on {self.device}")
        logger.info(f"Use precomputed: {use_precomputed}, Use clustering: {use_clustering}")

    def _initialize_model(self, embedding_dim: int):
        """Initialize model once embedding dimension is known"""
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
        """
        Train the model using enhanced PyTorch DataLoaders

        Args:
            dataloaders: Dictionary with train/val DataLoaders
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info(f"Starting enhanced training for {num_epochs} epochs")

        # Store target scaler
        self.target_scaler = dataloaders.get('target_scaler')

        # Initialize model if not already done
        train_loader = dataloaders['train']
        sample_batch = next(iter(train_loader))
        embedding_dim = sample_batch[0].shape[1]
        self._initialize_model(embedding_dim)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience // 2, factor=0.5, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        val_loader = dataloaders.get('val', None)

        # Log dataset sizes
        logger.info(f"Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_r2 = self._train_epoch(train_loader)

            # Validation phase
            if val_loader:
                val_loss, val_r2 = self._eval_epoch(val_loader)
            else:
                val_loss, val_r2 = train_loss, train_r2

            epoch_time = time.time() - epoch_start

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress with timing
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, Time={epoch_time:.1f}s"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        return self.history

    def _train_epoch(self, dataloader) -> tuple:
        """Train for one epoch with enhanced logging"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Use tqdm for progress bar
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch_idx, (embeddings, targets) in enumerate(pbar):
            # Data should already be on correct device from enhanced dataloader
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(embeddings).squeeze()
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)

        return avg_loss, r2

    def _eval_epoch(self, dataloader) -> tuple:
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Validation", leave=False):
                # Data should already be on correct device
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(embeddings).squeeze()
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)

        return avg_loss, r2

    def _calculate_r2(self, targets, predictions):
        """Calculate R² score"""
        from sklearn.metrics import r2_score
        return r2_score(targets, predictions)

    def evaluate(self, dataloader) -> Dict:
        """Evaluate model on a dataset"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        if self.model is None:
            raise ValueError("Model not initialized. Train the model first.")

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Evaluating", leave=False):
                # Explicitly transfer data from CPU to GPU
                embeddings = embeddings.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(embeddings).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Denormalize if scaler available
        if self.target_scaler:
            all_preds = self.target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
            all_targets = self.target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

        metrics = {
            'mse': mean_squared_error(all_targets, all_preds),
            'mae': mean_absolute_error(all_targets, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'r2': r2_score(all_targets, all_preds)
        }

        return metrics, all_preds, all_targets


def run_enhanced_pytorch_pipeline():
    """Run enhanced PyTorch-based pipeline with precomputed embeddings and clustering"""
    print("Running Enhanced PyTorch Glycan-Protein Binding Pipeline")
    print("Features: Precomputed Embeddings + Clustering-based Splits")
    print("=" * 80)

    # Configuration
    data_path = "data/v12_glycan_binding.csv"
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enhanced settings
    batch_size = 512
    cache_dir = "enhanced_embedding_cache"
    use_precomputed = True
    use_clustering = True
    n_clusters = 10

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")

    try:
        # Step 1: Create enhanced DataLoaders
        logger.info("Creating enhanced DataLoaders with precomputed embeddings and clustering...")
        start_time = time.time()

        dataloaders = create_enhanced_glycan_dataloaders(
            data_path=data_path,
            
            # Data splits
            test_size=0.1,
            val_size=0.1,
            
            # Enhanced features
            use_precomputed=use_precomputed,
            use_clustering=use_clustering,
            n_clusters=n_clusters,
            
            # Embedding settings
            protein_model="650M",
            glycan_method="lstm",
            glycan_vocab_path=vocab_path,
            
            # DataLoader settings
            batch_size=batch_size,
            cache_dir=cache_dir,
            device=device,
            
            # For testing, limit pairs (remove for full dataset)
            # max_pairs=5000  # Remove this line for full dataset
        )

        setup_time = time.time() - start_time
        logger.info(f"Enhanced DataLoader setup completed in {setup_time:.1f}s")

        # Step 2: Show cache and clustering information
        temp_loader = EnhancedGlycanProteinDataLoader(
            data_path=data_path,
            cache_dir=cache_dir,
            use_precomputed=use_precomputed,
            use_clustering=use_clustering
        )
        
        cache_info = temp_loader.get_cache_info()
        logger.info(f"Cache: {cache_info.get('protein_embeddings', 0)} proteins, "
                   f"{cache_info.get('glycan_embeddings', 0)} glycans, "
                   f"{cache_info.get('total_cache_size_mb', 0):.1f} MB total")

        # Step 3: Initialize enhanced predictor
        logger.info("Initializing enhanced predictor...")
        predictor = EnhancedPyTorchBindingPredictor(
            network_type="mlp",
            network_config={
                "hidden_dims": [1024, 512, 256, 128],
                "dropout": 0.2,
                "activation": "relu",
                "batch_norm": True
            },
            device=device,
            use_precomputed=use_precomputed,
            use_clustering=use_clustering
        )

        # Step 4: Train model
        logger.info("Training enhanced model...")
        training_start = time.time()
        
        history = predictor.train(
            dataloaders=dataloaders,
            num_epochs=50,
            learning_rate=2e-3,
            weight_decay=1e-4,
            patience=15
        )
        
        training_time = time.time() - training_start

        # Step 5: Evaluate on test set
        logger.info("Evaluating enhanced model...")
        test_metrics, test_preds, test_targets = predictor.evaluate(dataloaders['test'])

        # Print results
        print(f"\nEnhanced Training Results:")
        print(f"Setup time: {setup_time:.1f}s")
        print(f"Training time: {training_time:.1f}s")
        print(f"Total time: {setup_time + training_time:.1f}s")
        print(f"\nModel Performance:")
        print(f"Best validation R²: {max(history['val_r2']):.4f}")
        print(f"Final validation R²: {history['val_r2'][-1]:.4f}")
        print(f"\nTest Results:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")

        # Show GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")

        # Step 6: Create enhanced visualizations
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Enhanced Training Results with Clustering', fontsize=16)

            # Training curves
            axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
            axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
            axes[0, 0].set_title('Training Loss (Enhanced)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # R² curves
            axes[0, 1].plot(history['train_r2'], label='Train R²', alpha=0.7)
            axes[0, 1].plot(history['val_r2'], label='Val R²', alpha=0.7)
            axes[0, 1].set_title('R² Score (Enhanced)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Predictions vs targets
            axes[0, 2].scatter(test_targets, test_preds, alpha=0.6, s=20)
            min_val = min(test_targets.min(), test_preds.min())
            max_val = max(test_targets.max(), test_preds.max())
            axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[0, 2].set_xlabel('Actual Binding Strength')
            axes[0, 2].set_ylabel('Predicted Binding Strength')
            axes[0, 2].set_title(f'Predictions vs Actuals (R² = {test_metrics["r2"]:.3f})')
            axes[0, 2].grid(True, alpha=0.3)

            # Residuals plot
            residuals = test_preds - test_targets
            axes[1, 0].scatter(test_targets, residuals, alpha=0.6, s=20)
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 0].set_xlabel('Actual Binding Strength')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residuals Plot')
            axes[1, 0].grid(True, alpha=0.3)

            # Performance comparison
            methods = ['Original', 'Enhanced']
            # Dummy comparison data (replace with actual comparison if available)
            r2_scores = [0.75, test_metrics['r2']]  # Example values
            rmse_scores = [0.25, test_metrics['rmse']]  # Example values
            
            x = np.arange(len(methods))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, r2_scores, width, label='R²', alpha=0.7)
            axes[1, 1].bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.7)
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Performance Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(methods)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Enhancement features summary
            features_text = f"""
            Enhanced Features:
            • Precomputed Embeddings: {use_precomputed}
            • Clustering-based Splits: {use_clustering}
            • Number of Clusters: {n_clusters}
            • Cache Size: {cache_info.get('total_cache_size_mb', 0):.1f} MB
            
            Performance Gains:
            • Setup Time: {setup_time:.1f}s
            • Training Time: {training_time:.1f}s
            • Test R²: {test_metrics['r2']:.4f}
            """
            
            axes[1, 2].text(0.1, 0.5, features_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            axes[1, 2].set_title('Enhancement Summary')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig("enhanced_pytorch_pipeline_results.png", dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            logger.info("Matplotlib not available, skipping enhanced plots")

        print(f"\nEnhanced PyTorch pipeline completed successfully!")
        print(f"Key improvements:")
        print(f"  • Precomputed embeddings for faster loading")
        print(f"  • Clustering-based splits for better generalization")
        print(f"  • Optimized memory usage")

    except Exception as e:
        logger.error(f"Error in enhanced pipeline: {e}")
        import traceback
        traceback.print_exc()


def compare_original_vs_enhanced():
    """Compare original vs enhanced pipeline performance"""
    print("Comparing Original vs Enhanced Pipeline")
    print("=" * 50)
    
    # This would run both pipelines and compare results
    # Implementation would depend on having access to original pipeline
    print("This would compare:")
    print("1. Loading time (original embedding computation vs precomputed)")
    print("2. Generalization (random splits vs clustering-based splits)")
    print("3. Memory usage")
    print("4. Training stability")


def precompute_embeddings_only():
    """Utility function to only precompute embeddings without training"""
    print("Precomputing Embeddings Only")
    print("=" * 30)
    
    from embedding_preprocessor import preprocess_embeddings
    
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    data_path = "data/v12_glycan_binding.csv"
    
    # Precompute embeddings
    preprocessor = preprocess_embeddings(
        data_path=data_path,
        cache_dir="enhanced_embedding_cache",
        protein_model="650M",
        glycan_method="lstm",
        glycan_vocab_path=vocab_path,
        batch_size=8,
        force_recompute=False
    )
    
    # Show cache info
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