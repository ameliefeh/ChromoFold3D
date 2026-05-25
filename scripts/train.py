"""Train one fold of K-fold cross-validation.

Loop over `--fold 0..K-1` for the full sweep; see README.md.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from fp_gnn.dataset import FluorProteinDataset
from fp_gnn.lit_module import FluorLitModule, kfold_split
from fp_gnn.model import FPNet

DATA_ROOT = "data"


def main():
    parser = argparse.ArgumentParser()
    # Sweep dimension and log organisation.
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="If set, runs land in logs/<exp-name>/ instead of logs/.",
    )
    # K-fold split.
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--n-val",
        type=int,
        default=17,
        help="Validation-set size carved from the K-fold train pool.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seeds torch / numpy / random and the K-fold partition."
    )
    # Optimisation.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=150,
        help="Safety cap; EarlyStopping usually decides actual length.",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="EarlyStopping patience on val_loss."
    )
    # Model.
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Node embedding dim H in both MPNNs and the seq encoder.",
    )
    parser.add_argument(
        "--steps", type=int, default=3, help="Number of message-passing rounds in each MPNN."
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ds = FluorProteinDataset(root=DATA_ROOT)

    train_idx, valid_idx, test_idx = kfold_split(
        len(ds),
        k=args.n_folds,
        fold=args.fold,
        n_val=args.n_val,
        seed=args.seed,
    )
    train_ds = [ds[int(i)] for i in train_idx]
    val_ds = [ds[int(i)] for i in valid_idx]
    test_ds = [ds[int(i)] for i in test_idx]
    print(
        f"K={args.n_folds} fold={args.fold}: "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    net = FPNet(node_embedding_dim=args.hidden, num_message_steps=args.steps)
    lit = FluorLitModule(
        net=net,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    save_dir = f"logs/{args.exp_name}" if args.exp_name else "logs"
    run_name = f"fold{args.fold}"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=[
            CSVLogger(save_dir=save_dir, name=run_name),
            TensorBoardLogger(save_dir=save_dir, name=run_name),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{val_loss:.4f}"
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        ],
    )
    trainer.fit(lit)
    # ModelCheckpoint saves the lowest-val_loss weights; "best" resolves to that.
    trainer.test(lit, ckpt_path="best")

    csv_logger = next(lg for lg in trainer.loggers if isinstance(lg, CSVLogger))
    out_csv = Path(csv_logger.log_dir) / "test_predictions.csv"
    pd.DataFrame(lit.test_predictions).to_csv(out_csv, index=False)
    print(f"wrote {len(lit.test_predictions)} test predictions to {out_csv}")


if __name__ == "__main__":
    main()
