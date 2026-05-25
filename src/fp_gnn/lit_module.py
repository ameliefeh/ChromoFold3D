"""Lightning module, 5-fold data splitter, and z-score utilities.

This file contains the core training components used by `scripts/train.py`,
which provides the CLI wrapper around them.

Keeping this logic separate avoids pulling in argparse and CLI code when
running notebooks, tests, or `--predict-only` mode.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def kfold_split(
    n: int, k: int = 5, fold: int = 0, n_val: int = 17, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create K-fold (train, val, test) splits for n items.

    The data is first shuffled using the given seed, then divided into K folds
    (with sizes differing by at most one item). For each fold, that fold is used
    as the test set.

    From the remaining data, a small validation set of size `n_val` is taken
    (after reshuffling), and the rest is used for training.

    Across all K folds, every protein appears in the test set exactly once.
    """
    if not 0 <= fold < k:
        raise ValueError(f"fold {fold} out of range for k={k}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    starts = np.concatenate([[0], np.cumsum(fold_sizes)])
    test_idx = perm[starts[fold] : starts[fold + 1]]
    rest = np.concatenate([perm[: starts[fold]], perm[starts[fold + 1] :]])

    # Independent sub-shuffle so val changes from fold to fold
    rest = np.random.default_rng(seed * 1000 + fold + 1).permutation(rest)
    val_idx = rest[:n_val]
    train_idx = rest[n_val:]
    return train_idx, val_idx, test_idx


def compute_zscore_stats(
    targets: torch.Tensor, kdas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training split statistics; standard deviation is forced to be at least 1.0 to avoid division-by-zero issues."""
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0, unbiased=False).clamp_min(1.0)
    kda_mean = kdas.mean()
    kda_std = kdas.std(unbiased=False).clamp_min(1.0)
    return target_mean, target_std, kda_mean, kda_std


class FluorLitModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        train_dataset: list[Data],
        val_dataset: list[Data],
        test_dataset: list[Data],
        batch_size: int = 8,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr

        # Z-score stats computed on the train split only.
        ys = torch.cat([d.y for d in train_dataset], dim=0)
        kdas = torch.cat([d.kda for d in train_dataset], dim=0)
        target_mean, target_std, kda_mean, kda_std = compute_zscore_stats(ys, kdas)

        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("kda_mean", kda_mean)
        self.register_buffer("kda_std", kda_std)

        # Filled in by test_step; written to test_predictions.csv at the end of
        # training so we can plot predicted-vs-true scatters across folds.
        self.test_predictions: list[dict[str, str | float]] = []

    def _attach_kda_z(self, batch):
        batch.kda_z = (batch.kda - self.kda_mean) / self.kda_std

    def _zscore_targets(self, y):
        return (y - self.target_mean) / self.target_std

    def _denormalize(self, y_z):
        return y_z * self.target_std + self.target_mean

    def training_step(self, batch, batch_idx):
        self._attach_kda_z(batch)
        pred_z = self.net(batch)
        y_z = self._zscore_targets(batch.y)
        loss = F.mse_loss(pred_z, y_z)
        # on_epoch=True so the metrics.csv has one train_loss per epoch (mean over
        # batches), which lines up with val_loss for plotting.
        self.log("train_loss", loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def _eval_step(self, batch, prefix):
        self._attach_kda_z(batch)
        pred_z = self.net(batch)
        pred = self._denormalize(pred_z)
        diff = pred - batch.y
        mse = (diff**2).mean(dim=0)
        mae = diff.abs().mean(dim=0)
        # Z-scored MSE -- single scalar, what ModelCheckpoint / EarlyStopping monitor.
        y_z = self._zscore_targets(batch.y)
        loss = F.mse_loss(pred_z, y_z)
        self.log(f"{prefix}_loss", loss, batch_size=self.batch_size)
        self.log(f"{prefix}_mse_brightness", mse[0], batch_size=self.batch_size)
        self.log(f"{prefix}_mse_emission", mse[1], batch_size=self.batch_size)
        self.log(f"{prefix}_mae_brightness", mae[0], batch_size=self.batch_size)
        self.log(f"{prefix}_mae_emission", mae[1], batch_size=self.batch_size)
        return pred

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        pred = self._eval_step(batch, "test")
        # Stash per-protein (true, predicted) pairs for the final scatter plot.
        pred_np = pred.detach().cpu().numpy()
        y_np = batch.y.cpu().numpy()
        for code, yi, pi in zip(batch.pdb_code, y_np, pred_np):
            self.test_predictions.append(
                {
                    "pdb_code": code,
                    "y_brightness": float(yi[0]),
                    "y_emission": float(yi[1]),
                    "pred_brightness": float(pi[0]),
                    "pred_emission": float(pi[1]),
                }
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
