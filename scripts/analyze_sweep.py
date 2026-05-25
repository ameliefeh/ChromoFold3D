"""
Aggregate K-fold test metrics from a sweep and generate summary plots.

This script scans `logs/<exp-name>/fold<F>/version_*/metrics.csv`, selects the row containing
test metrics (i.e. where `test_*` values are present), and computes mean ± standard deviation
across folds. It also produces four figures saved to `<sweep_dir>/plots/`:

    loss_curves.png
        Train and validation loss vs epoch, averaged across folds with shaded std.

    pred_vs_true.png
        Predicted vs true scatter plots for both targets, including all folds.

    pred_vs_true_oof.png
        Out-of-fold (OOF) scatter plot (emission | brightness), with points coloured by
        spectral class (<470 nm blue, 470–530 green, 530–580 yellow, >580 red).
        Includes RMSE, MAE, and R² summary boxes.

    pred_vs_true_annotated.png
        Same OOF scatter plots, but with annotations for the 5 closest predictions (green)
        and 5 largest outliers (red) per panel.

Usage:
    python scripts/analyze_sweep.py --exp-name kfold5
"""

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Shared style ──────────────────────────────────────────────────────────────
_FS_TITLE = 13
_FS_LABEL = 11
_FS_TICK  = 9
_STYLE_APPLIED = False


def _apply_style():
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
    })
    _STYLE_APPLIED = True


_WL_CLASS_COLORS = [
    (470,  "#3377bb"),   # blue   — < 470 nm
    (530,  "#44aa44"),   # green  — 470–530 nm
    (580,  "#ddbb00"),   # yellow — 530–580 nm
    (None, "#cc3333"),   # red    — > 580 nm
]


def _class_colors(wavelengths: np.ndarray) -> list:
    """Categorical colour per emission-wavelength spectral class."""
    out = []
    for wl in wavelengths:
        for threshold, color in _WL_CLASS_COLORS:
            if threshold is None or wl < threshold:
                out.append(color)
                break
    return out


def _color_legend_handles(markersize: int = 7) -> list:
    entries = [
        ("#3377bb", "< 470 nm (blue)"),
        ("#44aa44", "470–530 nm (green)"),
        ("#ddbb00", "530–580 nm (yellow)"),
        ("#cc3333", "> 580 nm (red)"),
    ]
    return [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                      markersize=markersize, label=lbl)
        for c, lbl in entries
    ]


def _stat_box(ax, rmse: float, mae: float, r2: float, unit: str = ""):
    u = f" {unit}" if unit else ""
    txt = f"RMSE = {rmse:.2f}{u}\nMAE  = {mae:.2f}{u}\nR²   = {r2:.3f}"
    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#cccccc"),
        family="monospace",
    )

LOGS_ROOT = "logs"


def fold_run_dirs(root: Path):
    """Yield (fold_index, version_dir_with_metrics) pairs for every fold under root.

    Picks the most recently modified version directory per fold, so re-running a
    fold automatically uses the new results instead of version_0.
    """
    for fold_dir in sorted(root.glob("fold*")):
        try:
            fold = int(fold_dir.name[len("fold"):])
        except ValueError:
            continue
        versions = sorted(fold_dir.glob("version_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        vdir = next((v for v in versions if (v / "metrics.csv").exists()), None)
        if vdir:
            yield fold, vdir


def collect_test_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for fold, vdir in fold_run_dirs(root):
        df = pd.read_csv(vdir / "metrics.csv")
        test = df.dropna(subset=["test_mae_brightness"])
        if test.empty:
            continue
        r = test.iloc[0]
        rows.append(
            {
                "fold": fold,
                "mae_bright": r.test_mae_brightness,
                "mae_emis": r.test_mae_emission,
                "mse_bright": r.test_mse_brightness,
                "mse_emis": r.test_mse_emission,
            }
        )
    return pd.DataFrame(rows)


def plot_loss_curves(root: Path, save_path: Path):
    """Train + val loss vs epoch, averaged across folds with a shaded std band."""
    train_curves, val_curves = [], []
    for _, vdir in fold_run_dirs(root):
        df = pd.read_csv(vdir / "metrics.csv")
        train_curves.append(df.dropna(subset=["train_loss"]).groupby("epoch")["train_loss"].mean())
        val_curves.append(df.dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curves, color in [("train", train_curves, "C0"), ("val", val_curves, "C1")]:
        max_epoch = max(int(c.index.max()) for c in curves)
        epochs = np.arange(max_epoch + 1)
        # Pad each fold's series with NaN where it stopped early, then nan-aware mean/std.
        mat = np.full((len(curves), len(epochs)), np.nan)
        for i, c in enumerate(curves):
            mat[i, c.index.to_numpy()] = c.to_numpy()
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        ax.plot(epochs, mean, color=color, label=f"{label} (mean of {len(curves)} folds)")
        # Clip the lower band so log-y can render mean - std near zero.
        ax.fill_between(epochs, np.maximum(mean - std, 1e-3), mean + std, color=color, alpha=0.2)

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("z-scored MSE loss (log scale)")
    ax.set_title("FPNet loss curves (5-fold CV, mean ± std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_pred_vs_true(root: Path, save_path: Path):
    """Scatter of predicted vs true for both targets, with R^2 in the title."""
    parts = []
    for fold, vdir in fold_run_dirs(root):
        pred_csv = vdir / "test_predictions.csv"
        if not pred_csv.exists():
            continue
        df = pd.read_csv(pred_csv)
        df["fold"] = fold
        parts.append(df)
    if not parts:
        print("No test_predictions.csv found; re-run training to generate them.")
        return
    preds = pd.concat(parts, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, label in [
        (axes[0], "brightness", "Brightness"),
        (axes[1], "emission", "Emission (nm)"),
    ]:
        y = preds[f"y_{target}"]
        p = preds[f"pred_{target}"]
        # Color points by fold so the visual matches the K-fold structure.
        for fold, sub in preds.groupby("fold"):
            ax.scatter(
                sub[f"y_{target}"], sub[f"pred_{target}"], s=25, alpha=0.7, label=f"fold {fold}"
            )
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
        # R^2 (coefficient of determination), by hand to keep deps lean.
        ss_res = ((y - p) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"predicted {label}")
        ax.set_title(f"{label}    R² = {r2:.2f}")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def _load_oof_predictions(root: Path) -> "pd.DataFrame | None":
    parts = []
    for fold, vdir in fold_run_dirs(root):
        pred_csv = vdir / "test_predictions.csv"
        if not pred_csv.exists():
            continue
        df = pd.read_csv(pred_csv)
        df["fold"] = fold
        parts.append(df)
    if not parts:
        print("No test_predictions.csv found; re-run training to generate them.")
        return None
    return pd.concat(parts, ignore_index=True)


def plot_pred_vs_true_oof(root: Path, save_path: Path):
    """
    1 x 2 scatter of all OOF predictions (5 folds aggregated):
    emission wavelength (nm) | brightness.
    Both panels are colour-coded by spectral class of the true emission wavelength for each protein
    (<470 nm blue, 470–530 green, 530–580 yellow, >580 red).
    """
    preds = _load_oof_predictions(root)
    if preds is None:
        return

    _apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "FPNet — Predicted vs. True  (Out-of-Fold, all 5 folds)",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    emission_wl = preds["y_emission"].values
    pt_clrs     = _class_colors(emission_wl)

    targets = [
        ("y_emission",   "pred_emission",   "Emission Wavelength (nm)", "nm"),
        ("y_brightness", "pred_brightness", "Brightness",               ""),
    ]

    for ax, (y_col, p_col, label, unit) in zip(axes, targets):
        t = preds[y_col].values
        p = preds[p_col].values

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae  = float(np.mean(np.abs(p - t)))
        ss_r = np.sum((p - t) ** 2)
        ss_t = np.sum((t - t.mean()) ** 2)
        r2   = float(1 - ss_r / (ss_t + 1e-12))

        mn, mx = t.min(), t.max()
        margin = (mx - mn) * 0.04

        ax.scatter(t, p, c=pt_clrs, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([mn - margin, mx + margin],
                [mn - margin, mx + margin],
                "k--", lw=1.4, zorder=2)

        _stat_box(ax, rmse, mae, r2, unit=unit)

        ax.set_xlabel(f"True {label}",      fontsize=_FS_LABEL)
        ax.set_ylabel(f"Predicted {label}", fontsize=_FS_LABEL)
        ax.set_title(label, fontsize=_FS_LABEL, fontweight="bold", pad=8)
        ax.tick_params(labelsize=_FS_TICK)

    axes[1].legend(
        handles=_color_legend_handles(markersize=7),
        title="Spectral class", title_fontsize=9,
        fontsize=9, loc="lower right", framealpha=0.88,
        handletextpad=0.4, labelspacing=0.3, borderpad=0.5,
    )

    fig.tight_layout(w_pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def plot_pred_vs_true_annotated(root: Path, save_path: Path, n_annotate: int = 5):
    """
    Same 1 x 2 OOF scatter as plot_pred_vs_true_oof, with the N closest-
    predicted proteins labelled in green and the N most-outlier in red per panel.
    Both panels colour-coded by spectral class of the true emission wavelength
    (<470 nm blue, 470–530 green, 530–580 yellow, >580 red).
    """
    preds = _load_oof_predictions(root)
    if preds is None:
        return

    _apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "FPNet — Predicted vs. True  (Out-of-Fold) — Best & Worst Annotated",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    emission_wl = preds["y_emission"].values
    pt_clrs     = _class_colors(emission_wl)

    targets = [
        ("y_emission",   "pred_emission",   "Emission Wavelength (nm)", "nm"),
        ("y_brightness", "pred_brightness", "Brightness",               ""),
    ]

    for ax, (y_col, p_col, label, unit) in zip(axes, targets):
        t  = preds[y_col].values
        p  = preds[p_col].values
        ae = np.abs(p - t)

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae  = float(np.mean(ae))
        ss_r = np.sum((p - t) ** 2)
        ss_t = np.sum((t - t.mean()) ** 2)
        r2   = float(1 - ss_r / (ss_t + 1e-12))

        mn, mx = t.min(), t.max()
        margin = (mx - mn) * 0.04

        ax.scatter(t, p, c=pt_clrs, s=45, alpha=0.65,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([mn - margin, mx + margin],
                [mn - margin, mx + margin],
                "k--", lw=1.4, zorder=2)

        _stat_box(ax, rmse, mae, r2, unit=unit)

        sorted_idx = np.argsort(ae)
        best_idx   = sorted_idx[:n_annotate]
        worst_idx  = sorted_idx[-n_annotate:]
        u_str      = f" {unit}" if unit else ""

        def _annotate(idx, color, _t=t, _p=p, _ae=ae, _mn=mn, _mx=mx):
            code = preds.iloc[idx]["pdb_code"]
            xi, yi = _t[idx], _p[idx]
            dy = (_mx - _mn) * 0.08 * (1 if (yi - xi) >= 0 else -1)
            ax.annotate(
                f"{code}\n|Δ|={_ae[idx]:.2f}{u_str}",
                xy=(xi, yi), xytext=(xi, yi + dy),
                fontsize=8, color=color, fontweight="bold",
                ha="center", va="bottom" if dy > 0 else "top",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                shrinkA=0, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.85, ec=color, lw=0.8),
                zorder=6,
            )
            ax.scatter([xi], [yi], c=color, s=80, zorder=7,
                       edgecolors="white", linewidths=0.6)

        for i in best_idx:
            _annotate(i, "#1a7a1a")   # dark green
        for i in worst_idx:
            _annotate(i, "#b80000")   # dark red

        ax.set_xlabel(f"True {label}",      fontsize=_FS_LABEL)
        ax.set_ylabel(f"Predicted {label}", fontsize=_FS_LABEL)
        ax.set_title(label, fontsize=_FS_LABEL, fontweight="bold", pad=8)
        ax.tick_params(labelsize=_FS_TICK)

    annotation_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#1a7a1a",
                      markersize=7, label=f"Top {n_annotate} closest predicted"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#b80000",
                      markersize=7, label=f"Top {n_annotate} most outlier"),
    ]
    axes[1].legend(
        handles=_color_legend_handles(markersize=7) + annotation_handles,
        title="Spectral class / annotation", title_fontsize=9,
        fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1),
        framealpha=0.95, edgecolor="#cccccc",
        handletextpad=0.4, labelspacing=0.3, borderpad=0.6,
    )

    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(right=0.82)  # make room for the external legend
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-name",
        default=None,
        help=f"Sweep dir under {LOGS_ROOT}/.  Default: {LOGS_ROOT}/ itself.",
    )
    args = ap.parse_args()

    root = Path(LOGS_ROOT) / args.exp_name if args.exp_name else Path(LOGS_ROOT)
    df = collect_test_metrics(root)
    if df.empty:
        print(f"No fold runs with test metrics found under {root}/.")
        return

    print(f"# Per-fold runs ({root})")
    print(df.sort_values("fold").to_string(index=False))
    print()
    print(f"# Aggregate across {len(df)} folds")
    print(f"  MAE brightness: {df.mae_bright.mean():6.2f} ± {df.mae_bright.std():.2f}")
    print(f"  MAE emission:   {df.mae_emis.mean():6.2f} ± {df.mae_emis.std():.2f}")

    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_loss_curves(root, plots_dir / "loss_curves.png")
    plot_pred_vs_true(root, plots_dir / "pred_vs_true.png")
    plot_pred_vs_true_oof(root, plots_dir / "pred_vs_true_oof.png")
    plot_pred_vs_true_annotated(root, plots_dir / "pred_vs_true_annotated.png")


if __name__ == "__main__":
    main()
