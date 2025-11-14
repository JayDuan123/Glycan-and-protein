"""
Neural Network Architectures for Protein-Glycan Binding Strength Prediction
Defines various neural network architectures for supervised learning on embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for binding strength regression
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3,
                 activation: str = "relu",
                 batch_norm: bool = True,
                 output_activation: Optional[str] = None):
        """
        Initialize MLP regressor

        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            batch_norm: Whether to use batch normalization
            output_activation: Output activation ('sigmoid', 'tanh', None)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Choose activation function
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        self.activation = activation_map.get(activation, nn.ReLU())

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        # Output activation if specified
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Predicted binding strength of shape (batch_size, 1)
        """
        return self.network(x)


class ResidualMLPRegressor(nn.Module):
    """
    MLP with residual connections for improved training
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 dropout: float = 0.3,
                 activation: str = "relu"):
        """
        Initialize Residual MLP

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension (constant across layers)
            num_layers: Number of residual blocks
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        self.activation = activation_map.get(activation, nn.ReLU())

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, self.activation)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.input_proj(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block for ResidualMLP"""

    def __init__(self, hidden_dim: int, dropout: float, activation):
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentionRegressor(nn.Module):
    """
    Attention-based regressor that learns to focus on important embedding features
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize Attention Regressor

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Output layers
        self.output_layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Predicted binding strength of shape (batch_size, 1)
        """
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch_size, hidden_dim)

        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Apply attention layers
        for attention, layer_norm, ffn in zip(self.attention_layers, self.layer_norms, self.ffns):
            # Self-attention with residual connection
            attn_out, _ = attention(x, x, x)
            x = layer_norm(x + attn_out)

            # Feed-forward with residual connection
            ffn_out = ffn(x)
            x = layer_norm(x + ffn_out)

        # Remove sequence dimension and apply output layers
        x = x.squeeze(1)  # (batch_size, hidden_dim)

        return self.output_layers(x)


class EnsembleRegressor(nn.Module):
    """
    Ensemble of multiple regressors for improved performance
    """

    def __init__(self,
                 input_dim: int,
                 models_config: List[dict],
                 ensemble_method: str = "mean"):
        """
        Initialize Ensemble Regressor

        Args:
            input_dim: Input embedding dimension
            models_config: List of model configurations
            ensemble_method: How to combine predictions ("mean", "weighted", "stacking")
        """
        super().__init__()

        self.ensemble_method = ensemble_method
        self.models = nn.ModuleList()

        # Create individual models
        for config in models_config:
            model_type = config.pop('type')
            if model_type == 'mlp':
                model = MLPRegressor(input_dim, **config)
            elif model_type == 'residual_mlp':
                model = ResidualMLPRegressor(input_dim, **config)
            elif model_type == 'attention':
                model = AttentionRegressor(input_dim, **config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.models.append(model)

        # Weights for weighted ensemble
        if ensemble_method == "weighted":
            self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))

        # Stacking layer for stacking ensemble
        elif ensemble_method == "stacking":
            self.stacking_layer = nn.Sequential(
                nn.Linear(len(self.models), len(self.models) // 2),
                nn.ReLU(),
                nn.Linear(len(self.models) // 2, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        # Get predictions from all models
        predictions = torch.stack([model(x) for model in self.models], dim=-1)

        if self.ensemble_method == "mean":
            return predictions.mean(dim=-1)

        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.weights, dim=0)
            return (predictions * weights).sum(dim=-1)

        elif self.ensemble_method == "stacking":
            # Flatten predictions for stacking layer
            predictions_flat = predictions.view(predictions.size(0), -1)
            return self.stacking_layer(predictions_flat)


class BindingStrengthNetworkFactory:
    """Factory class for creating binding strength prediction networks"""

    @staticmethod
    def create_network(network_type: str, input_dim: int, **kwargs) -> nn.Module:
        """
        Create a network for binding strength prediction

        Args:
            network_type: Type of network ('mlp', 'residual_mlp', 'attention', 'ensemble')
            input_dim: Input embedding dimension
            **kwargs: Additional arguments for the network

        Returns:
            Neural network model
        """
        if network_type == "mlp":
            return MLPRegressor(input_dim, **kwargs)

        elif network_type == "residual_mlp":
            return ResidualMLPRegressor(input_dim, **kwargs)

        elif network_type == "attention":
            return AttentionRegressor(input_dim, **kwargs)

        elif network_type == "ensemble":
            return EnsembleRegressor(input_dim, **kwargs)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

    @staticmethod
    def get_default_config(network_type: str) -> dict:
        """Get default configuration for a network type"""
        configs = {
            "mlp": {
                "hidden_dims": [512, 256, 128],
                "dropout": 0.3,
                "activation": "relu",
                "batch_norm": True
            },
            "residual_mlp": {
                "hidden_dim": 512,
                "num_layers": 4,
                "dropout": 0.3,
                "activation": "relu"
            },
            "attention": {
                "hidden_dim": 256,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1
            },
            "ensemble": {
                "models_config": [
                    {"type": "mlp", "hidden_dims": [512, 256, 128], "dropout": 0.3},
                    {"type": "residual_mlp", "hidden_dim": 512, "num_layers": 3},
                    {"type": "attention", "hidden_dim": 256, "num_heads": 8}
                ],
                "ensemble_method": "mean"
            }
        }

        return configs.get(network_type, {})


if __name__ == "__main__":
    # Example usage
    input_dim = 2560  # Example dimension for concatenated embeddings

    # Test different network architectures
    networks = {
        "MLP": MLPRegressor(input_dim),
        "Residual MLP": ResidualMLPRegressor(input_dim),
        "Attention": AttentionRegressor(input_dim),
    }

    # Test with random input
    batch_size = 32
    x = torch.randn(batch_size, input_dim)

    for name, network in networks.items():
        output = network(x)
        print(f"{name}: Input {x.shape} -> Output {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in network.parameters())}")

    # Test ensemble
    ensemble_config = BindingStrengthNetworkFactory.get_default_config("ensemble")
    ensemble = BindingStrengthNetworkFactory.create_network("ensemble", input_dim, **ensemble_config)
    ensemble_output = ensemble(x)
    print(f"Ensemble: Input {x.shape} -> Output {ensemble_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in ensemble.parameters())}")