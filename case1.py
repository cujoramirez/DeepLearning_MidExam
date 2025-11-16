from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

COLUMNS = [
    "EID",
    "AbsT",
    "RelT",
    "NID",
    "Temp",
    "RelH",
    "L1",
    "L2",
    "Occ",
    "Act",
    "Door",
    "Win",
]
FEATURE_COLUMNS = ["Temp", "RelH", "L1", "L2"]
DERIVED_COLUMNS = [f"{col}_delta" for col in FEATURE_COLUMNS]
MODEL_FEATURES = FEATURE_COLUMNS + DERIVED_COLUMNS
TARGET_COLUMN = "Temp_next"
SEQ_LEN = 48
INPUT_DIM = len(MODEL_FEATURES)


def load_room_climate_data(files: Sequence[Path]) -> pd.DataFrame:
    """Load and concatenate the measurement CSV files."""
    frames = [pd.read_csv(path, header=None, names=COLUMNS) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("AbsT").reset_index(drop=True)
    df[TARGET_COLUMN] = df["Temp"].shift(-1)
    df = df.dropna(subset=[TARGET_COLUMN])
    for col in FEATURE_COLUMNS:
        delta = df[col].diff().fillna(0.0)
        df[f"{col}_delta"] = delta
    return df

def temporal_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe chronologically."""

    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


class TemperatureDataset(Dataset):
    """Simple PyTorch dataset wrapping numpy arrays."""
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self) -> int: 
        return len(self.features)
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:  
        return self.features[idx], self.targets[idx]


class SequenceTemperatureDataset(Dataset):
    """Dataset that returns (seq_len, feature_dim) tensors for sequence models."""
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int) -> None:
        seq_features, seq_targets = create_sequence_arrays(features, targets, seq_len)
        self.features = torch.tensor(seq_features, dtype=torch.float32)
        self.targets = torch.tensor(seq_targets, dtype=torch.float32)
    def __len__(self) -> int: 
        return len(self.features)
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]: 
        return self.features[idx], self.targets[idx]


def create_sequence_arrays(
    features: np.ndarray, targets: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if len(features) != len(targets):
        raise ValueError("Features and targets must share the same length")
    if len(features) < seq_len:
        raise ValueError("Not enough samples to build at least one sequence")
    seq_features: List[np.ndarray] = []
    seq_targets: List[np.ndarray] = []
    for idx in range(seq_len - 1, len(features)):
        seq_features.append(features[idx - seq_len + 1 : idx + 1])
        seq_targets.append(targets[idx])
    return np.stack(seq_features), np.array(seq_targets)

class TemperatureRegressor(nn.Module):
    """Feed-forward regressor with configurable hidden layers."""

    def __init__(
        self, input_dim: int, hidden_sizes: Sequence[int], dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor: 
        return self.network(x).squeeze(-1)


class TemperatureSequenceRegressor(nn.Module):
    """LSTM-based regressor that consumes sliding windows of sensor readings."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        head_hidden: int | None = None,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        head_hidden = head_hidden or max(hidden_size // 2, 32)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class TrainingConfig:
    name: str
    architecture: str = "mlp"
    hidden_sizes: Tuple[int, ...] = (128, 64, 32)
    lstm_hidden: int = 128
    lstm_layers: int = 2
    seq_len: int = SEQ_LEN
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 60
    patience: int = 20


@dataclass
class TrainingResult:
    name: str
    best_val_loss: float
    history: Dict[str, List[float]]
    state_dict_path: Path
    config: TrainingConfig


class Trainer:
    def __init__(
        self,
        train_dataset: TemperatureDataset,
        val_dataset: TemperatureDataset,
        device: torch.device,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

    def train(self, config: TrainingConfig, save_dir: Path) -> TrainingResult:
        save_dir.mkdir(parents=True, exist_ok=True)
        train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False
        )

        model = self._build_model(config).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0
        history = {"train": [], "val": []}

        for epoch in range(1, config.epochs + 1):
            train_loss = self._run_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            history["train"].append(train_loss)
            history["val"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config.patience:
                break

        state_path = save_dir / f"{config.name.replace(' ', '_').lower()}_model.pt"
        if best_state is None:
            best_state = model.state_dict()
        torch.save(best_state, state_path)

        return TrainingResult(
            name=config.name,
            best_val_loss=best_val,
            history=history,
            state_dict_path=state_path,
            config=config,
        )

    def _build_model(self, config: TrainingConfig) -> nn.Module:
        arch = config.architecture.lower()
        if arch == "lstm":
            return TemperatureSequenceRegressor(
                input_dim=INPUT_DIM,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                dropout=config.dropout,
            )
        if arch == "mlp":
            return TemperatureRegressor(INPUT_DIM, config.hidden_sizes, config.dropout)
        raise ValueError(f"Unknown architecture: {config.architecture}")

    def _run_epoch(
        self,
        model: TemperatureRegressor,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        model.train()
        running_loss = 0.0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(features)
        return running_loss / len(loader.dataset)

    def _evaluate(
        self, model: TemperatureRegressor, loader: DataLoader, criterion: nn.Module
    ) -> float:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                predictions = model(features)
                loss = criterion(predictions, targets)
                running_loss += loss.item() * len(features)
        return running_loss / len(loader.dataset)

def evaluate_model(
    model: TemperatureRegressor,
    loader: DataLoader,
    device: torch.device,
    scaler_y: StandardScaler,
) -> Dict[str, float]:
    model.eval()
    preds: List[float] = []
    actuals: List[float] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            outputs = model(features).cpu().numpy()
            preds.extend(outputs)
            actuals.extend(targets.numpy())

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).ravel()

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, preds, actuals


def plot_losses(history: Dict[str, List[float]], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predictions(actual: np.ndarray, preds: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, preds, alpha=0.5, s=10)
    min_val, max_val = actual.min(), actual.max()
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Actual vs Predicted Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_predictions.png", dpi=300)
    plt.close()

    residuals = actual - preds
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=40, alpha=0.7, edgecolor="black")
    plt.xlabel("Prediction Error (°C)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=300)
    plt.close()


def build_datasets(
    df: pd.DataFrame,
    seq_len: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset, StandardScaler, StandardScaler]:
    train_df, val_df, test_df = temporal_train_val_test_split(df)
    scaler_X = StandardScaler().fit(train_df[MODEL_FEATURES])
    scaler_y = StandardScaler().fit(train_df[[TARGET_COLUMN]])

    def transform(split: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = scaler_X.transform(split[MODEL_FEATURES])
        targets = scaler_y.transform(split[[TARGET_COLUMN]]).ravel()
        return features, targets

    train_X, train_y = transform(train_df)
    val_X, val_y = transform(val_df)
    test_X, test_y = transform(test_df)

    if seq_len is not None:
        train_ds: Dataset = SequenceTemperatureDataset(train_X, train_y, seq_len)
        val_ds: Dataset = SequenceTemperatureDataset(val_X, val_y, seq_len)
        test_ds: Dataset = SequenceTemperatureDataset(test_X, test_y, seq_len)
    else:
        train_ds = TemperatureDataset(train_X, train_y)
        val_ds = TemperatureDataset(val_X, val_y)
        test_ds = TemperatureDataset(test_X, test_y)

    return train_ds, val_ds, test_ds, scaler_X, scaler_y


def hyperparameter_candidates(base_epochs: int, seq_len: int) -> List[TrainingConfig]:
    return [
        TrainingConfig(
            name="LSTM_Base",
            architecture="lstm",
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=64,
            epochs=base_epochs,
            patience=15,
            seq_len=seq_len,
        ),
        TrainingConfig(
            name="LSTM_Deeper",
            architecture="lstm",
            lstm_hidden=192,
            lstm_layers=3,
            dropout=0.35,
            lr=8e-4,
            weight_decay=2e-5,
            batch_size=64,
            epochs=base_epochs,
            patience=18,
            seq_len=seq_len,
        ),
        TrainingConfig(
            name="LSTM_WiderBatch",
            architecture="lstm",
            lstm_hidden=256,
            lstm_layers=2,
            dropout=0.25,
            lr=1.2e-3,
            weight_decay=1e-5,
            batch_size=96,
            epochs=base_epochs,
            patience=15,
            seq_len=seq_len,
        ),
    ]


def run_experiment(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_files = [
        Path("Room-Climate-Datasets/datasets-location_C/room_climate-location_C-measurement32.csv"),
        Path("Room-Climate-Datasets/datasets-location_C/room_climate-location_C-measurement33.csv"),
    ]
    df = load_room_climate_data(data_files)

    train_ds, val_ds, test_ds, scaler_X, scaler_y = build_datasets(df, seq_len=args.seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(train_ds, val_ds, device)

    outputs = Path("problem1_outputs")
    outputs.mkdir(exist_ok=True)

    baseline_config = TrainingConfig(
        name="Baseline_LSTM",
        architecture="lstm",
        lstm_hidden=160,
        lstm_layers=2,
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=64,
        epochs=args.epochs,
        patience=18,
        seq_len=args.seq_len,
    )
    baseline_result = trainer.train(baseline_config, outputs)
    plot_losses(
        baseline_result.history,
        "Baseline Training vs Validation Loss",
        outputs / "baseline_loss.png",
    )

    # Hyperparameter tuning phase
    tuning_results: List[TrainingResult] = []
    configs = hyperparameter_candidates(args.tune_epochs, args.seq_len)
    for config in configs:
        result = trainer.train(config, outputs / "tuning")
        tuning_results.append(result)
        plot_losses(
            result.history,
            f"{config.name} Loss Curves",
            outputs / "tuning" / f"{config.name.replace(' ', '_').lower()}_loss.png",
        )

    tuning_summary = [
        {"name": r.name, "best_val_loss": r.best_val_loss, **asdict(r.config)}
        for r in tuning_results
    ]
    (outputs / "tuning").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_summary).to_csv(outputs / "tuning" / "tuning_results.csv", index=False)

    best_result = min(tuning_results, key=lambda r: r.best_val_loss)

    # Final training with best config for more epochs
    final_config = TrainingConfig(
        name="Final",
        architecture=best_result.config.architecture,
        hidden_sizes=best_result.config.hidden_sizes,
        lstm_hidden=best_result.config.lstm_hidden,
        lstm_layers=best_result.config.lstm_layers,
        dropout=best_result.config.dropout,
        lr=best_result.config.lr,
        weight_decay=best_result.config.weight_decay,
        batch_size=best_result.config.batch_size,
        epochs=args.final_epochs,
        patience=20,
        seq_len=args.seq_len,
    )
    final_result = trainer.train(final_config, outputs / "final")
    plot_losses(
        final_result.history,
        "Final Model Loss Curves",
        outputs / "final" / "final_loss.png",
    )

    # Load best final model for evaluation
    final_model = trainer._build_model(final_config).to(device)
    final_model.load_state_dict(torch.load(final_result.state_dict_path, map_location=device))

    test_loader = DataLoader(test_ds, batch_size=final_config.batch_size, shuffle=False)
    metrics, preds, actuals = evaluate_model(final_model, test_loader, device, scaler_y)
    plot_predictions(np.array(actuals), np.array(preds), outputs / "final")

    with open(outputs / "final" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Final test metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper():<4}: {value:.4f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Room temperature prediction pipeline")
    parser.add_argument("--epochs", type=int, default=60, help="Baseline training epochs")
    parser.add_argument(
        "--tune-epochs", type=int, default=40, help="Epochs per hyperparameter run"
    )
    parser.add_argument(
        "--final-epochs", type=int, default=120, help="Epochs for final training"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help="Number of past timesteps used as context for sequence models",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Enable a lightweight configuration for rapid sanity checks",
    )
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 15)
        args.tune_epochs = min(args.tune_epochs, 10)
        args.final_epochs = min(args.final_epochs, 30)
        args.seq_len = min(args.seq_len, 12)
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(cli_args)
