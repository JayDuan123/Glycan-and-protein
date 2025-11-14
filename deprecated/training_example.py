"""
PyTorch Integration Example: Glycan-Protein Binding Prediction
Shows how to use the PyTorch DataLoader with the binding strength predictor
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
import logging

# Import our custom modules
from dataloader.glycan_dataloader import create_glycan_dataloaders
from Integrated_Embedder import GlycanProteinPairEmbedder
from binding_strength_networks import BindingStrengthNetworkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchBindingPredictor:
    """
    PyTorch-based binding strength predictor using GPU-optimized DataLoaders
    """

    def __init__(self,
                 embedder: GlycanProteinPairEmbedder,
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,
                 device: Optional[str] = None):
        """
        Initialize the predictor

        Args:
            embedder: Pre-initialized embedder
            network_type: Type of neural network
            network_config: Network configuration
            device: Computing device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder

        # Get embedding dimension
        self.embedding_dim = embedder.get_output_dim()

        # Create network
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)
        self.model = BindingStrengthNetworkFactory.create_network(
            network_type, self.embedding_dim, **self.network_config
        ).to(self.device)

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

        logger.info(f"Initialized predictor on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self,
              dataloaders: Dict[str, torch.utils.data.DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10) -> Dict:
        """
        Train the model using PyTorch DataLoaders

        Args:
            dataloaders: Dictionary with train/val DataLoaders
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Store target scaler
        self.target_scaler = dataloaders.get('target_scaler')

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience//2, factor=0.5, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        train_loader = dataloaders['train']
        val_loader = dataloaders.get('val', None)

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_r2 = self._train_epoch(train_loader)

            # Validation phase
            if val_loader:
                val_loss, val_r2 = self._eval_epoch(val_loader)
            else:
                val_loss, val_r2 = train_loss, train_r2

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

            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}"
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for embeddings, targets in dataloader:
            # Data is already on GPU from DataLoader
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
            for embeddings, targets in dataloader:
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

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for embeddings, targets in dataloader:
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

    def predict_from_dataloader(self, dataloader) -> np.ndarray:
        """Make predictions using a DataLoader"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for embeddings, _ in dataloader:
                outputs = self.model(embeddings).squeeze()
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions)

        # Denormalize if scaler available
        if self.target_scaler:
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions


def run_pytorch_pipeline():
    """Run complete PyTorch-based pipeline"""
    print("Running PyTorch Glycan-Protein Binding Pipeline")
    print("=" * 60)

    # Configuration
    data_path = "../data/v12_glycan_binding.csv"
    vocab_path = "../GlycanEmbedder_Package/glycoword_vocab.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = GlycanProteinPairEmbedder(
            protein_model="650M",
            protein_model_dir="resources/esm-model-weights",
            glycan_method="lstm",
            glycan_vocab_path=vocab_path,
            fusion_method="concat",
            device=device
        )

        # Create DataLoaders
        logger.info("Creating DataLoaders...")
        dataloaders = create_glycan_dataloaders(
            data_path=data_path,
            embedder=embedder,
            test_size=0.2,
            val_size=0.1,
            batch_size=32,
            max_pairs=1000,  # Limit for testing
            device=device
        )

        # Initialize predictor
        logger.info("Initializing predictor...")
        predictor = PyTorchBindingPredictor(
            embedder=embedder,
            network_type="mlp",
            network_config={
                "hidden_dims": [512, 256, 128],
                "dropout": 0.3,
                "activation": "relu"
            },
            device=device
        )

        # Train model
        logger.info("Training model...")
        history = predictor.train(
            dataloaders=dataloaders,
            num_epochs=30,
            learning_rate=1e-3,
            patience=10
        )

        # Evaluate on test set
        logger.info("Evaluating model...")
        test_metrics, test_preds, test_targets = predictor.evaluate(dataloaders['test'])

        # Print results
        print(f"\nTraining completed!")
        print(f"Final validation R²: {history['val_r2'][-1]:.4f}")
        print(f"Test Results:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")

        # Plot results if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Training curves
            axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
            axes[0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # R² curves
            axes[1].plot(history['train_r2'], label='Train R²', alpha=0.7)
            axes[1].plot(history['val_r2'], label='Val R²', alpha=0.7)
            axes[1].set_title('R² Score')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('R²')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Predictions vs targets
            axes[2].scatter(test_targets, test_preds, alpha=0.6, s=20)
            min_val = min(test_targets.min(), test_preds.min())
            max_val = max(test_targets.max(), test_preds.max())
            axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[2].set_xlabel('Actual Binding Strength')
            axes[2].set_ylabel('Predicted Binding Strength')
            axes[2].set_title(f'Predictions vs Actuals (R² = {test_metrics["r2"]:.3f})')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("pytorch_pipeline_results.png", dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            logger.info("Matplotlib not available, skipping plots")

        print("\nPyTorch pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


def test_dataloader_only():
    """Test just the DataLoader functionality"""
    print("Testing PyTorch DataLoader only")

    try:
        # Mock embedder for testing
        class MockEmbedder:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            def get_output_dim(self):
                return 128

            def embed_pairs(self, pairs, batch_size=32, return_numpy=False):
                n_pairs = len(pairs)
                embedding_dim = 128
                embeddings = torch.randn(n_pairs, embedding_dim)
                return embeddings.to(self.device) if not return_numpy else embeddings.numpy()

        embedder = MockEmbedder()

        # Test DataLoader creation
        dataloaders = create_glycan_dataloaders(
            data_path="../data/v12_glycan_binding.csv",
            embedder=embedder,
            batch_size=16,
            max_pairs=100
        )

        # Test iteration
        print("Testing DataLoader iteration:")
        train_loader = dataloaders['train']

        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: {embeddings.shape}, {targets.shape}")
            print(f"  Devices: {embeddings.device}, {targets.device}")

            if batch_idx >= 2:
                break

        print("DataLoader test successful!")

    except Exception as e:
        print(f"DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Choose which test to run
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dataloader":
        test_dataloader_only()
    else:
        run_pytorch_pipeline()