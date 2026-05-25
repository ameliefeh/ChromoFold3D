# ChromoFold 3D - Fluorescent Protein Property Prediction

Prediction tool for peak emission wavelength and brightness of fluorescent proteins from its 3D structure (PDB), molecular weight (kDa), and amino-acid
sequence.

## Encoding 
The model (`FPNet`) pools three feature streams:

- a protein residue-contact graph (Cα within 8 Å) → MPNN
- a chromophore atom graph (CCD-typed bonds + 3D distances) → MPNN
- a 21-dim sequence vector (amino acids/residues composition + log-length) → small MLP

These three are concatenated with the standardised kDa (size of the protein) and fed to a small
MLP head that outputs `[brightness, emission]`.

Data (PDB files for each protein) are in `data/raw/`; Library code lives in `src/fp_gnn/`; CLI scripts are in `scripts/`. Documentation containing detailed explanations of the code, methodology and analysis is in `docs/DOCUMENTATION.md`

## Setup

**Requirements**: Python >= 3.11 and the [uv](https://docs.astral.sh/uv/) package manager.

**Step 1 — Install uv** (skip if you already have it):

On Mac/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 2 — Install dependencies** from the repo root.

Choose the option that matches your hardware — and use the same `--extra` flag every time you run the code:

```bash
uv sync --extra cpu   # Mac, no NVIDIA GPU, or unsure → start here
uv sync --extra gpu   # NVIDIA GPU with CUDA driver 580+ → faster training
```

**Step 3 — Activate the virtual environment** that uv just created:

On Mac/Linux:
```bash
source .venv/bin/activate
```

```bash
uv pip install -e .
```

On Windows:
```powershell
.venv\Scripts\activate
```

Your terminal prompt will show `(.venv)` once it is active. You need to do this once per terminal session before running any scripts.

## Run

**Step 1 — Build graphs** from `data/raw/*.pdb` → `data/processed/data.pt` (one-time, ~minutes). Re-run only when `labels.csv` or the PDB files change.

```bash
uv run python scripts/process.py
```

**Step 2 — Train one fold** of 5-fold cross-validation. EarlyStopping (patience=20) ends the run when `val_loss` plateaus. Replace `--extra gpu` with `--extra cpu` if you are not using a GPU.

On Mac/Linux:
```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra cpu python scripts/train.py --fold 0
```
On Windows:
```powershell
$env:CUDA_VISIBLE_DEVICES=0; uv run --extra gpu python scripts/train.py --fold 0
```

**Step 3 — Full 5-fold sweep** → aggregate → save report figures. Replace `--extra gpu` with `--extra cpu` if you are not using a GPU.

On Mac/Linux:
```bash
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 uv run --extra cpu python scripts/train.py \
    --fold $fold --exp-name kfold5
done
```
On Windows:
```powershell
foreach ($fold in 0,1,2,3,4) {
  $env:CUDA_VISIBLE_DEVICES=0; uv run --extra gpu python scripts/train.py --fold $fold --exp-name kfold5
}
```

Then on both:
```bash
uv run python scripts/analyze_sweep.py --exp-name kfold5
```

**N.B: kfold5/ folder in logs/ folder are the folds we runned and from where we got our data. It constitutes an experiment of 5 different folds (fold folder inside kfold5/). You can run your own experiment by running --exp-name your_name with your own folds inside it. Keep in mind that analyze_sweep.py is plotting the results for the most recent version_/ inside each of your fold/ folder and aggregate them. In this repo, you have the version_0 from each of our fold that constitutes the results data.**

**Step 4 — Live curves** in browser (open in a separate terminal):

```bash
uv run tensorboard --logdir logs
# -> http://localhost:6006
```

### Output files 
`scripts/analyze_sweep.py` prints per-fold MAE and the mean ± std across
folds, and writes four PNGs into `logs/<exp-name>/plots/`:

| File | Description |
|---|---|
| `loss_curves.png` | Train/val loss vs epoch, mean of 5 folds with shaded ±1σ |
| `pred_vs_true.png` | Predicted vs true scatter for both targets, R² in title |
| `pred_vs_true_oof.png` | Out-of-fold scatter (emission \| brightness) with protein points colour-coded by spectral class (<470 nm blue, 470–530 green, 530–580 yellow, >580 red) and a RMSE/MAE/R² stat box per panel |
| `pred_vs_true_annotated.png` | Same scatter with the 5 closest-predicted proteins labelled in green and the 5 most-outlier in red per panel |

Logs land in `logs/[<exp-name>/]fold<F>/version_*/`:
- `metrics.csv`: per-step / per-epoch losses, val and test metrics.
- `test_predictions.csv`: one row per held-out protein with true and
  predicted (brightness, emission). Used by `--plots`.
- `events.out.tfevents.*`: same metrics for TensorBoard.
- `checkpoints/best-*.ckpt`: lowest-`val_loss` weights, used at test time.

**We highly recommend you to read the `docs/DOCUMENTATION.md` for further explanations of each piece of code, methodology, processing, training, validation, test and analysis. :)**

### Web interface

```bash
streamlit run app.py
```

Opens a browser UI with two tabs:

- **Prediction** — upload any fluorescent protein PDB file, enter its molecular weight (kDa), and get predicted emission wavelength and brightness. The page also shows how the demo model performs on its held-out test proteins.
- **K-Fold Results** — per-fold MAE/RMSE table and the four figures generated by `scripts/analyze_sweep.py`.

The app loads the best checkpoint from fold 3 (lowest `val_loss` across all folds). Run `scripts/process.py` and at least `scripts/train.py --fold 3 --exp-name kfold5` before launching.

## Project Structure

```
ChromoFold 3D/
├── src/fp_gnn/                        # library code (importable)
│   ├── pdb_io.py                      # parse PDB files: residue list + Cα coords, chromophore block
│   ├── protein_graph.py               # build the residue-contact PyG Data
│   ├── chromophore_graph.py           # build the chromophore atom PyG Data via RDKit
│   ├── dataset.py                     # FluorProteinDataset (PyG InMemoryDataset wrapper)
│   │                                  #   + sequence_to_features (AA composition + log-length)
│   ├── model.py                       # ChromMPNN + ProteinMPNN + FPNet
│   └── lit_module.py                  # FluorLitModule (Lightning) + kfold_split
│
├── scripts/                           # CLI entry points (no library code)
│   ├── process.py                     # run the dataset build once
│   ├── train.py                       # one fold of K-fold CV
│   └── analyze_sweep.py               # aggregate fold metrics, save report figures
│
├── docs/
│   └── DOCUMENTATION.md               # full documentation of methods, training, and analysis
│
├── data/
│   ├── raw/                           # 119 PDB files (1HUY is skipped at processing time)
│   ├── labels.csv                     # 120 rows -> 118 usable proteins
│   ├── ccd_cache/                     # cached CCD SMILES per chromophore code
│   └── processed/                     # cached PyG dataset
│
├── logs/                              # training outputs
│   └── kfold5/                        # 5-fold cross-validation runs
│       ├── fold{0..4}/                # per-fold checkpoints and metrics
│       └── plots/                     # aggregated figures (loss curves, pred vs true)
│
├── app.py                             # Streamlit web interface (prediction + K-fold results)
├── pyproject.toml                     # project metadata and dependencies (uv)
├── uv.lock                            # locked dependency versions
```

## Data

- `data/raw/*.pdb`: 119 fluorescent-protein structures (1HUY is skipped
  at processing time, leaving 118).
- `data/labels.csv`: per-protein scalars and targets (kDa, brightness,
  emission, quantum yield, sequence).
- `data/ccd_cache/` (gitignored): cached CCD SMILES per chromophore code.
- `data/processed/` (gitignored): cached PyG dataset after first run.

**N.B: data/ccd_cache/ and data/processed/ are created by the script process.py. You can delete those and just keep the raw/*.pdb and labels.csv, as the cached data are created by running the process.py script ONCE. You do not need to run this script for every fold you will run with train.py, unless you change the dataset of proteins**

## Configuration 

All hyperparameters are set via CLI flags in [`scripts/train.py`](scripts/train.py), so they can be overriden without editing the file. Default values reflect the configuration used to produce the results in the report.

**Data split**

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `--n-folds` | `5` | Number of folds for K-fold cross-validation |
| `--n-val` | `17` | Validation-set size carved from the K-fold train pool |
| `--seed` | `0` | Seeds torch / numpy / random and the K-fold partition |

**Training**

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `--batch-size` | `8` | Mini-batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--max-epochs` | `150` | Safety cap on training length (EarlyStopping usually stops earlier) |
| `--patience` | `20` | EarlyStopping patience on `val_loss` |

**Model architecture**

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `--hidden` | `64` | Node embedding dimension `H` in both MPNNs and the sequence encoder |
| `--steps` | `3` | Number of message-passing rounds in each MPNN |

