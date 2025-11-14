import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Integrated_Embedder import GlycanProteinPairEmbedder
from binding_strength_networks import BindingStrengthNetworkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BindingStrengthPredictor:
    def __init__(self,
                 protein_model: str = "650M",
                 protein_model_dir: str = "resources/esm-model-weights",
                 glycan_method: str = "lstm",
                 glycan_vocab_path: Optional[str] = None,
                 glycan_hidden_dims: Optional[List[int]] = None,
                 glycan_readout: str = "mean",
                 fusion_method: str = "concat",
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,
                 device: Optional[str] = None,
                 random_seed: int = 42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = GlycanProteinPairEmbedder(
            protein_model=protein_model,
            protein_model_dir=protein_model_dir,
            glycan_method=glycan_method,
            glycan_vocab_path=glycan_vocab_path,
            glycan_hidden_dims=glycan_hidden_dims,
            glycan_readout=glycan_readout,
            fusion_method=fusion_method,
            device=self.device
        )
        self.embedding_dim = self.embedder.get_output_dim()
        self.network_type = network_type
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)
        self.model = BindingStrengthNetworkFactory.create_network(
            network_type, self.embedding_dim, **self.network_config
        ).to(self.device)
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def prepare_data(self,
                     pairs: List[Tuple[str, str]],
                     strengths: List[float],
                     batch_size: int = 32,
                     val_split: float = 0.2,
                     test_split: float = 0.1,
                     normalize_targets: bool = True) -> Dict[str, DataLoader]:
        embeddings = self.embedder.embed_pairs(pairs, batch_size=batch_size, return_numpy=True)
        strengths = np.array(strengths)
        if normalize_targets:
            strengths = self.scaler.fit_transform(strengths.reshape(-1, 1)).flatten()
        X = torch.FloatTensor(embeddings)
        y = torch.FloatTensor(strengths)
        dataset = TensorDataset(X, y)
        n_samples = len(dataset)
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_test - n_val
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

    def train(self,
              data_loaders: Dict[str, DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10,
              min_delta: float = 1e-4,
              scheduler_config: Optional[Dict] = None) -> Dict:
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
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
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        for epoch in range(num_epochs):
            train_loss, train_metrics = self._train_epoch(data_loaders['train'])
            val_loss, val_metrics = self._eval_epoch(data_loaders['val'])
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        self.is_fitted = True
        return self.training_history

    def _train_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_preds)
        return avg_loss, metrics

    def _eval_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_preds)
        return avg_loss, metrics

    def _calculate_metrics(self, targets: List[float], predictions: List[float]) -> Dict:
        targets = np.array(targets)
        predictions = np.array(predictions)
        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions)
        }

    def predict(self,
                pairs: List[Tuple[str, str]],
                batch_size: int = 32,
                return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        embeddings = self.embedder.embed_pairs(pairs, batch_size=batch_size, return_numpy=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size].to(self.device)
                pred = self.model(batch).squeeze()
                predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        if hasattr(self.scaler, 'scale_'):
            predictions_np = predictions.cpu().numpy().reshape(-1, 1)
            predictions_np = self.scaler.inverse_transform(predictions_np).flatten()
            predictions = torch.FloatTensor(predictions_np)
        return predictions.cpu().numpy() if return_numpy else predictions

    def evaluate(self, data_loader: DataLoader) -> Dict:
        _, metrics = self._eval_epoch(data_loader)
        return metrics

    def plot_training_history(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(self.training_history['train_loss'])
        axes[0, 0].plot(self.training_history['val_loss'])
        axes[0, 1].plot([m['r2'] for m in self.training_history['train_metrics']])
        axes[0, 1].plot([m['r2'] for m in self.training_history['val_metrics']])
        axes[1, 0].plot([m['mae'] for m in self.training_history['train_metrics']])
        axes[1, 0].plot([m['mae'] for m in self.training_history['val_metrics']])
        axes[1, 1].plot([m['rmse'] for m in self.training_history['train_metrics']])
        axes[1, 1].plot([m['rmse'] for m in self.training_history['val_metrics']])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, path: str):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'network_type': self.network_type,
            'network_config': self.network_config,
            'embedding_dim': self.embedding_dim,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        torch.save(save_dict, path)

    def load_model(self, path: str):
        save_dict = torch.load(path, map_location=self.device)
        self.network_type = save_dict['network_type']
        self.network_config = save_dict['network_config']
        self.embedding_dim = save_dict['embedding_dim']
        self.model = BindingStrengthNetworkFactory.create_network(
            self.network_type, self.embedding_dim, **self.network_config
        ).to(self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.scaler = save_dict['scaler']
        self.training_history = save_dict['training_history']
        self.is_fitted = save_dict['is_fitted']


def load_data_from_file(file_path: str) -> Tuple[List[Tuple[str, str]], List[float]]:
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        pairs = [(item['glycan_iupac'], item['protein_sequence']) for item in data]
        strengths = [item['binding_strength'] for item in data]
    elif file_path.suffix.lower() in ['.csv', '.tsv']:
        sep = ',' if file_path.suffix.lower() == '.csv' else '\t'
        df = pd.read_csv(file_path, sep=sep)
        pairs = list(zip(df['glycan_iupac'], df['protein_sequence']))
        strengths = df['binding_strength'].tolist()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    return pairs, strengths


if __name__ == "__main__":
    vocab_path = "../GlycanEmbedder_Package/glycoword_vocab.pkl"
    pairs = [
        ("Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
         "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("Neu5Ac(a2-3)Gal(b1-4)Glc",
         "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP")
    ] * 50
    np.random.seed(42)
    strengths = np.random.normal(0.5, 0.2, len(pairs)).tolist()
    predictor = BindingStrengthPredictor(
        protein_model="650M",
        protein_model_dir="../resources/esm-model-weights",
        glycan_method="lstm",
        glycan_vocab_path=vocab_path,
        fusion_method="concat",
        network_type="mlp",
        network_config={
            "hidden_dims": [512, 256, 128],
            "dropout": 0.3,
            "activation": "relu"
        }
    )
    data_loaders = predictor.prepare_data(
        pairs, strengths,
        batch_size=16,
        val_split=0.2,
        test_split=0.1
    )
    history = predictor.train(
        data_loaders,
        num_epochs=20,
        learning_rate=1e-3,
        patience=5,
        scheduler_config={'type': 'reduce_on_plateau', 'patience': 3, 'factor': 0.5}
    )
    predictor.evaluate(data_loaders['test'])
    predictor.predict(pairs[:5])
    predictor.plot_training_history()
    predictor.save_model("binding_strength_model.pth")
