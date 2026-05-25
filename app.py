"""
Streamlit web interface for ChromoFold 3D — Fluorescent Protein Property Prediction.

Run with:
    streamlit run app.py

Requires:
  - Processed dataset:  uv run python scripts/process.py
  - At least fold 3:    uv run python scripts/train.py --fold 3 --exp-name kfold5
  - Plots (optional):   uv run python scripts/analyze_sweep.py --exp-name kfold5
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
import torch

from fp_gnn.chromophore_graph import build_chromophore_graph
from fp_gnn.dataset import FPData, FluorProteinDataset, sequence_to_features
from fp_gnn.lit_module import FluorLitModule, kfold_split
from fp_gnn.model import FPNet
from fp_gnn.pdb_io import get_protein_residue_ca
from fp_gnn.protein_graph import build_protein_graph

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs" / "kfold5"

N_FOLDS = 5
DEMO_FOLD = 3   # fold with the lowest val_loss (0.3156) among all trained folds
N_VAL = 17
SEED = 0
HIDDEN = 64
STEPS = 3

# Maps PDB 3-letter residue names to single-letter codes for sequence_to_features.
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_best_checkpoint(fold: int) -> Path | None:
    """Return the checkpoint with the lowest val_loss for the given fold."""
    candidates = sorted(
        (LOGS_ROOT / f"fold{fold}").glob("version_*/checkpoints/best-*.ckpt")
    )
    if not candidates:
        return None

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("val_loss=")[1])
        except (IndexError, ValueError):
            return float("inf")

    return min(candidates, key=_loss)


@st.cache_resource
def load_model_and_dataset():
    """Load dataset + demo fold checkpoint once, cache for the session."""
    processed = DATA_ROOT / "processed" / "data.pt"
    if not processed.exists():
        return None, None, None, (
            "Processed dataset not found. "
            "Run `uv run python scripts/process.py` first."
        )

    ds = FluorProteinDataset(root=str(DATA_ROOT))
    train_idx, val_idx, test_idx = kfold_split(
        len(ds), k=N_FOLDS, fold=DEMO_FOLD, n_val=N_VAL, seed=SEED
    )
    train_ds = [ds[int(i)] for i in train_idx]
    val_ds   = [ds[int(i)] for i in val_idx]
    test_ds  = [ds[int(i)] for i in test_idx]

    ckpt_path = _find_best_checkpoint(DEMO_FOLD)
    if ckpt_path is None:
        return None, ds, None, (
            f"No checkpoint found for fold {DEMO_FOLD}. "
            f"Run `uv run python scripts/train.py --fold {DEMO_FOLD} --exp-name kfold5` first."
        )

    net = FPNet(node_embedding_dim=HIDDEN, num_message_steps=STEPS)
    lit = FluorLitModule(
        net=net,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"])
    lit.eval()
    return lit, ds, ckpt_path, None


def _predict(pdb_bytes: bytes, kda: float, lit: FluorLitModule):
    """Build graphs from an uploaded PDB and run FPNet inference.

    Returns (result_dict, error_str). Exactly one of them is None.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fh:
        fh.write(pdb_bytes)
        tmp = Path(fh.name)
    try:
        res_names, ca = get_protein_residue_ca(tmp)
        if not res_names:
            return None, "No residues found in chain A of the uploaded PDB."

        prot  = build_protein_graph(res_names, ca, cutoff=8.0)
        chrom = build_chromophore_graph(tmp)

        seq       = "".join(_AA3_TO_1.get(r, "X") for r in res_names)
        seq_feat  = sequence_to_features(seq)

        data = FPData(
            x=prot.x,
            edge_index=prot.edge_index,
            edge_attr=prot.edge_attr,
            chrom_x=chrom.x,
            chrom_edge_index=chrom.edge_index,
            chrom_edge_attr_chem=chrom.edge_attr_chem,
            chrom_edge_attr_dist=chrom.edge_attr_dist,
            kda=torch.tensor([kda], dtype=torch.float),
            y=torch.tensor([[0.0, 0.0]], dtype=torch.float),
            pdb_code="query",
            seq_feat=seq_feat,
        )
        data.kda_z = (data.kda - lit.kda_mean) / lit.kda_std

        with torch.no_grad():
            pred_z = lit.net(data)
            pred   = lit._denormalize(pred_z)

        brightness, emission = pred[0].tolist()
        return {
            "emission_nm": emission,
            "brightness": brightness,
            "sequence": seq,
            "n_residues": len(res_names),
        }, None

    except Exception as exc:
        return None, str(exc)
    finally:
        tmp.unlink(missing_ok=True)


def _collect_fold_metrics() -> pd.DataFrame:
    rows = []
    for fold in range(N_FOLDS):
        for vdir in sorted((LOGS_ROOT / f"fold{fold}").glob("version_*"), reverse=True):
            mc = vdir / "metrics.csv"
            if not mc.exists():
                continue
            df   = pd.read_csv(mc)
            test = df.dropna(subset=["test_mae_brightness"])
            if test.empty:
                continue
            r = test.iloc[0]
            rows.append({
                "Fold": fold,
                "MAE emission (nm)": f"{r.test_mae_emission:.2f}",
                "MAE brightness":    f"{r.test_mae_brightness:.2f}",
                "RMSE emission (nm)": f"{np.sqrt(r.test_mse_emission):.2f}",
                "RMSE brightness":    f"{np.sqrt(r.test_mse_brightness):.2f}",
            })
            break  # use highest-version run that has test metrics
    return pd.DataFrame(rows)


def _run_on_test_set(lit: FluorLitModule, ds: FluorProteinDataset) -> pd.DataFrame:
    """Run demo model on its held-out test proteins and return a comparison table."""
    _, _, test_idx = kfold_split(len(ds), k=N_FOLDS, fold=DEMO_FOLD, n_val=N_VAL, seed=SEED)
    rows = []
    with torch.no_grad():
        for i in test_idx:
            sample = ds[int(i)]
            sample.kda_z = (sample.kda - lit.kda_mean) / lit.kda_std
            pred_z = lit.net(sample)
            pred   = lit._denormalize(pred_z)
            br_pred, em_pred = pred[0].tolist()
            br_true, em_true = sample.y[0].tolist()
            rows.append({
                "PDB code":            sample.pdb_code,
                "True emission (nm)":  f"{em_true:.0f}",
                "Pred emission (nm)":  f"{em_pred:.1f}",
                "True brightness":     f"{br_true:.1f}",
                "Pred brightness":     f"{br_pred:.1f}",
                "|Δ| emission (nm)":   f"{abs(em_pred - em_true):.1f}",
            })
    return pd.DataFrame(rows)


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="ChromoFold 3D", layout="wide")
st.title("ChromoFold 3D — Fluorescent Protein Property Prediction")
st.markdown(
    "Predict **emission wavelength** and **brightness** from a fluorescent protein's "
    "3D structure (PDB file) and molecular weight."
)

lit, ds, ckpt_path, load_error = load_model_and_dataset()

if load_error:
    st.error(load_error)
    st.stop()

tab_pred, tab_kfold = st.tabs(["Prediction", "K-Fold Results"])

# ── Tab: Prediction ───────────────────────────────────────────────────────────

with tab_pred:
    col_in, col_out = st.columns(2)

    with col_in:
        st.subheader("Input")
        pdb_file = st.file_uploader(
            "Upload a fluorescent protein PDB file",
            type=["pdb"],
            help="The PDB must contain chain A residues with Cα atoms and a HETATM chromophore block.",
        )
        kda = st.number_input(
            "Molecular weight (kDa)",
            min_value=1.0,
            max_value=500.0,
            value=26.0,
            step=0.1,
        )

        st.caption(
            f"Demo model: fold {DEMO_FOLD} best checkpoint — `{ckpt_path.name}`"
        )

        predict_clicked = st.button(
            "Predict",
            disabled=(pdb_file is None),
            type="primary",
        )

        if predict_clicked and pdb_file is not None:
            with st.spinner("Building protein & chromophore graphs, running FPNet..."):
                result, error = _predict(pdb_file.read(), kda, lit)
            if error:
                st.error(f"Prediction failed: {error}")
            else:
                st.success("Prediction complete")
                m1, m2 = st.columns(2)
                m1.metric("Emission Wavelength", f"{result['emission_nm']:.1f} nm")
                m2.metric("Brightness",          f"{result['brightness']:.3f}")
                with st.expander("Show details"):
                    st.write(f"**Residues parsed from chain A:** {result['n_residues']}")
                    st.text_area(
                        "Derived single-letter sequence",
                        value=result["sequence"],
                        height=100,
                        disabled=True,
                    )

        if pdb_file is None:
            st.info(
                "Upload a PDB file above to get started.  \n"
                "Any file from `data/raw/` works — e.g. `7ZCT.pdb` (mScarlet3, 592 nm)."
            )

    with col_out:
        st.subheader(f"Demo model — fold {DEMO_FOLD} test-set proteins")
        st.caption(
            "These are the ~24 held-out proteins the demo checkpoint never saw during training."
        )
        df_test = _run_on_test_set(lit, ds)
        st.dataframe(df_test, use_container_width=True, hide_index=True)

# ── Tab: K-Fold Results ───────────────────────────────────────────────────────

with tab_kfold:
    st.subheader("5-Fold Cross-Validation Results")
    st.markdown(
        "Run `uv run python scripts/train.py --fold <0..4> --exp-name kfold5` for all folds, "
        "then `uv run python scripts/analyze_sweep.py --exp-name kfold5` to generate plots."
    )

    df_metrics = _collect_fold_metrics()
    if not df_metrics.empty:
        st.markdown("**Per-fold test metrics (best checkpoint per fold):**")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        mae_em = df_metrics["MAE emission (nm)"].astype(float).values
        mae_br = df_metrics["MAE brightness"].astype(float).values
        st.markdown(
            f"**Aggregate ({len(df_metrics)} folds):** "
            f"MAE emission = **{mae_em.mean():.2f} ± {mae_em.std():.2f} nm**  |  "
            f"MAE brightness = **{mae_br.mean():.2f} ± {mae_br.std():.2f}**"
        )
    else:
        st.info("No test metrics yet — train all 5 folds first.")

    plots = [
        ("Loss Curves — train / val (mean ± std across folds)", "loss_curves.png"),
        ("Predicted vs. True — per fold",                       "pred_vs_true.png"),
        ("Predicted vs. True — Out-of-Fold (spectral class colour)", "pred_vs_true_oof.png"),
        ("Predicted vs. True — Annotated (best & worst per target)", "pred_vs_true_annotated.png"),
    ]
    any_plot = False
    for title, fname in plots:
        path = LOGS_ROOT / "plots" / fname
        if path.exists():
            any_plot = True
            st.markdown(f"**{title}**")
            st.image(str(path))
    if not any_plot:
        st.info(
            "No plots found. Run `uv run python scripts/analyze_sweep.py --exp-name kfold5`."
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## How it works")
st.sidebar.markdown("""
**Input**
- A fluorescent protein PDB file (chain A, with chromophore HETATM record)
- Molecular weight in kDa

**Three feature streams fed to FPNet:**
1. Protein residue-contact graph (Cα pairs within 8 Å) → MPNN
2. Chromophore atom graph (CCD-typed bonds + 3D distances) → MPNN
3. Amino-acid composition + log-length → MLP

All three are concatenated with the standardised kDa and passed to an MLP head that outputs `[brightness, emission]`.

**Targets**
- Emission wavelength (nm) — reliably predicted (~28 nm MAE over 234 nm range)
- Brightness — near-baseline; depends on very local chromophore environment that Cα-level structure doesn't encode well

**Training**
- 118 fluorescent proteins from RCSB PDB
- 5-fold cross-validation, EarlyStopping (patience 20)
- Adam lr=1e-3, hidden=64, 3 message-passing steps
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## Supported chromophores")
st.sidebar.markdown(
    "CRO, CRQ, CR2, CR7, CR8, CRF, DYG, GYS, GYC, CCY, CFY, "
    "NRQ, NRP, OFM, OIM, CH6, CH7, WCR, NYG, BJF, SWG, CSH, "
    "PIA, BLR, B2H, FAD, FMN, 5SQ, 4M9"
)
