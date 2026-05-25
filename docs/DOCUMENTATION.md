## SUMMARY_DOCUMENTATION

## What is our dataset ?

Our dataset for the 3D model contains `119 fluorescent proteins` from which we extracted the 3D coordinates from Protein Data Bank (PDB) `data/labels.csv` has, for each, kDa, brightness, peak emission (nm), quantum yield, and amino-acid sequence. Each fluorescent protein has a beta-barrel shape from its residues, and inside its core has a ligand, that is called a 'chromophore'. This ligand is covalently bounded to the protein and is responsible for the fluorescence properties, i.e. like the color the protein emits (`emission wavelenght`) or the intensity of the color (`brightness`).

When we run the process.py script to process the dataset, the script filters proteins that have missing structural 3D coordinates. Therefore, we have a final dataset of `118 fluorescent proteins`.

## A. PIPELINE OF THE 3D PROJECT

For each protein, we are interested in predicting the brightness and the peak emission wavelength from the 3D structure. The brightness tells us how bright the protein is. The emission wavelenght tells us which color does the protein emit. As features, we also add the molecular weight (kdA) and the sequence of the protein (amino acid composition). 

Pipeline:

```
data/raw/*.pdb  +  data/labels.csv
       |
       v   scripts/process.py          (one-time, ~minutes)
data/processed/data.pt        <-- per-protein PyG Data: protein graph,
                                   chromophore graph, sequence vector,
                                   targets
       |
       v   scripts/train.py --fold N   (5 times, one per fold)
logs/<exp>/foldN/version_*/   <-- metrics.csv, test_predictions.csv,
                                   checkpoints/best-*.ckpt
       |
       v   scripts/analyze_sweep.py
logs/<exp>/plots/*.png        <-- four report figures
```

Core ideas:

1. **Two graphs, one head.** A single architecture (FPNet) with three feature streams pooled to a per-protein vector constitutes our model: we have a protein-level MPNN over residue contacts and an atom-level MPNN over the chromophore that are pooled to per-protein vectors, concatenated with sequence features and standardised kDa (size of the protein), and run through a small MLP to predict (brightness, emission).
2. **K-fold CV.** With a dataset of only N=118 proteins, a single 70/15/15 split (about 83 train, 18 validation, and 17 test samples) is often too small to give a reliable estimate of model performance. At this scale, every individual protein has a large impact on the evaluation metrics, so the results can vary significantly depending on which samples happen to fall into the test set. This makes both validation and test performance noisy and sensitive to random splitting, and it also wastes valuable data by permanently holding out a portion of the dataset from training. Using 5-fold cross-validation addresses this issue by rotating the test set so that every protein is used exactly once for testing and multiple times for training, producing a more stable and less variance-prone estimate of performance. Each fold has a completely different test set (non-overlapping). Instead of relying on a single potentially unrepresentative split, K-fold CV averages results across multiple folds, making it much more robust for small biological datasets where data is limited and each sample carries substantial weight.
3. **Early stopping for validation.** Instead of training each five model from the k-fold on a fixed number of epochs, we use Early stopping to pick the optimal number of epochs for each model.  We use `ModelCheckpoint` of Pytorch Lightning to save the
lowest-`val_loss` weights; `trainer.test(ckpt_path="best")` runs the final test pass against those, not the last-epoch weights.
4. **Getting our prediction for each protein** Our loss function minimizes the brightness and wavelenght with equal weight for both. The global prediction is the union of all fold test predictions (out-of-fold prediction).

## B. QUICK EXPLANATIONS OF OUR 3 CLI SCRIPTS IN SCRIPTS/ TO USE

### `scripts/process.py`

This script is responsible for building the final processed dataset that the model trains on. It reads raw input PDB files from the data/raw/ directory along with the labels.csv file, and converts them into graph-based training samples.

Internally, it runs all preprocessing steps defined in the dataset pipeline (protein parsing, chromophore extraction create the ccd_cache folder in data/, feature construction, etc. See the C section below that explains the src/), and then packages everything into a single cached PyTorch Geometric dataset file: data/processed/data.pt.

This step only needs to be RUN ONCE, unless the raw data or labels are changed (for example if we want to add more proteins). Because the script has hardcoded paths (like DATA_ROOT = "data"), it does not require command-line arguments and is intended as a simple “build dataset” utility

### `scripts/train.py`

This is the main training entry point for running a single fold of 5-fold cross-validation. Each execution of this script trains and evaluates the model on one specific split of the data. By default, it reproduces the exact experimental setup used in the reported results (the defaults reproduce the runs reported in this documentation in D section or also written in the README), but it is also flexible: you can choose which fold to run using the --fold argument (from 0 to 4), and you can group multiple runs under a shared experiment name using --exp-name, which organizes outputs under logs/exp_name/. (for us it is called kfold5)

The kfold5 is the exp-name we used for our data, which when you launched each fold with `train.py`, you create the `folds` folder in `logs/kfold5`. Each fold contains one run (version_0) that corresponds to our data for each fold. Rerunning the train.py script under the same exp-name will create other versions. By default, the recent versions in each fold is used to plot the aggregated data.

Each 5-fold splits the dataset in train/validation/test, details are explained in the section `src/fp_gnn/lit_module.py` below. We implemented an early stopping for each fold so that we can have an optimized number of epochs with weights that corresponds to the one that minimizes the val loss. Other parameters have been varied but just manually and we kept as defaults the best ones. We implemented this k-fold method because we have a small dataset, but a very sophisticated model, so like this we can reduce considerably the variance of our model and have more meaningful predictions.

The hyperparameters that actually matter are also CLI flags so you can override them without editing the file: `--seed`, `--lr`,
`--batch-size`, `--max-epochs`, `--patience`, `--hidden`, `--steps`, `--n-folds`, `--n-val`. Defaults are the values used in the report Adam, lr 1e-3, batch size 8, max 150 epochs (also written in our README). `EarlyStopping(patience=20)` on val_loss
decides the actual length of the number of epochs for each fold; `ModelCheckpoint(monitor="val_loss")` picks the weights tested.

During execution, the script creates both a CSV logger and a TensorBoard logger. Each fold is stored in its own directory under logs/[exp-name]/fold<F>/version_*. These directories contain training metrics, model checkpoints (specifically the best-performing model), and after testing, a file called test_predictions.csv, which stores per-protein predictions and ground truth values for later analysis.

### `scripts/analyze_sweep.py`

This script is used after all folds have been trained. It aggregates results across multiple training runs and produces summary statistics and visualizations. It scans through all experiment directories under `logs/<exp-name>/fold<F>/version_*/metrics.csv`, extracts validation and test metrics from each run and the checkpoint (optimized nb of epochs and corresponding weights), and computes per-fold MAE values and reports the overall cross-validation mean and standard deviation. By default, it will run the newest version from each fold. In this repo, we only have the version_0 for each fold, which lead to our data. If you run fold again with the same experiment name, you will create new versions.

The script also loads the saved test_predictions.csv files from each fold. Using these, it generates the general figures in `logs/<exp-name>/plots/`

- `loss_curves.png`: which shows how training and validation loss evolve over epochs. These curves are plotted on a logarithmic scale and aggregated across folds, with a shaded region indicating variability (±1 standard deviation).
- `pred_vs_true.png` : compares predicted versus actual values for both brightness and emission. Each fold is shown in a different color, and the overall quality of fit is quantified using an R² score.
- `pred_vs_true_oof.png` : compares predicted versus actual values for both brightness and emission but by aggregating all the folds (5 folds, OOF). Each protein data point is colored based on their emission color, and we report RMSE, MAE and R² score.
- `pred_vs_true_annotated.png`: same as the _oof.png but here we labeled the proteins that are very well predicted (on the line) and those that are outliers.

## C. QUICK EXPLANATIONS OF EACH MODULE .py FILE IN SRC/

### `src/fp_gnn/pdb_io.py`

This module contains PDB_file-parsers that extract the structural information (3D coordinates of residues and chromophore) from each protein file .pdb that were retrieved from Protein Data Bank. 
- The function `get_chromophore_code(pdb_path) -> str` returns the first residue name that matches a predefined set of known chromophore codes. This set is manually curated from the structures present in the dataset, and it must be updated whenever new chromophore types are introduced.
- The function `get_chromophore_pdb_block(pdb_path, residue_code) -> str` extracts all 3D coordinates of the atoms corresponding to a given chromophore residue in chain A and assembles them into a minimal PDB block suitable for RDKit processing. The chromophore is converted to a mol object, that we can afterwards use for the graph builder like we saw in the exercices in class.
- The function `get_protein_residue_ca(pdb_path) -> (list[str], np.ndarray)` extracts a simplified protein backbone representation by parsing a list of residue names alongside the corresponding Cα coordinate array from the pdb file. For simplicity, we only keep coordinates of chain A for each protein. For each residue, we extract only the Calpha atom, as the Calpha atoms represent the overall protein backbone.

### `src/fp_gnn/protein_graph.py`

This module constructs a residue-level protein graph from Cα coordinates and amino acid identities. The function `build_protein_graph(residue_names, ca_coords, cutoff=8.0) -> Data` produces a PyTorch Geometric Data object where each node corresponds to a residue and is represented as a 20-dimensional one-hot encoding of the amino acid type.

Edges are formed between all residue pairs whose Cα–Cα distance is within an 8 Å cutoff, and each edge is bidirectional with no self-loops. The edge attributes store the exact Euclidean distance between residues as a single scalar feature. Node positions are also preserved in the pos attribute to support geometric or spatially aware models. The cutoff is configurable but we kept it as 8 Å throughout experimentation.

- `x`: `[N, 20]` one-hot of the amino acid type (`STANDARD_AA` lookup);
  non-standard residues become a zero row with a `[WARN]` print.
- `edge_index`: `[2, E]` long, both directions, no self-loops, every
  pair within 8 Å Cα-Cα.
- `edge_attr`: `[E, 1]` float, the actual distance for each edge.
- `pos`: `[N, 3]` Cα coordinates.

### `src/fp_gnn/chromophore_graph.py`

This module constructs a atom-level chromophore graph from the atom coordinates of each chromophore in each protein and using chemical information as well. We highly inspired ourselves from the Graph Neural Network and MPNN exercices in the Class (we also used RDKIT but with slightly improved atom and edges features for the atom). 
Particularly, the function `build_chromophore_graph(pdb_path) -> Data`:
1. Find the chromophore residue code (`get_chromophore_code`) and the
   PDB block for that residue (`get_chromophore_pdb_block`).
2. Look up the canonical CCD SMILES for that code via the RCSB REST
   API, with a local JSON cache in `data/ccd_cache/` (so we hit the
   network at most once per code).
3. Build an RDKit `Mol` from the PDB block (no bond orders, just heavy
   atoms in 3D), then assign bond orders by matching against the SMILES
   template (`AssignBondOrdersFromTemplate`).
4. Convert the sanitised `Mol` to a PyG `Data` (`mol_to_graph`) using
   OGB atom/bond featurisers from RDKIT.

### `src/fp_gnn/dataset.py`

This module defines both feature engineering utilities and the dataset construction pipeline. The function `sequence_to_features(seq) -> [1, 21] tensor` converts a protein sequence into a compact 21-dimensional feature vector consisting of amino acid composition fractions for the 20 standard amino acids plus a logarithmic encoding of sequence length. This representation is intentionally simple and deterministic, complementing experimentally measured molecular weight (kDa) with a lightweight sequence-derived signal.

The `FluorProteinDataset` (a PyG `InMemoryDataset`) is responsible for constructing the full dataset from the labels.csv file in data/. On first initialization, it iterates through all entries, constructs protein and chromophore graphs for each valid PDB, and skips any structures listed in SKIP_PDB_CODES. The processed dataset is then serialized into a single file (data.pt) for fast reuse in subsequent runs, so that we do not need to repeat the graph construction.

Each sample is stored as an FPData object containing both protein-level and chromophore-level graph representations, along with features such as sequence features, molecular weight, and target values for brightness and emission.

Each `FPData` carries:

```
x, edge_index, edge_attr            # protein graph
chrom_x, chrom_edge_index,          # chromophore graph
chrom_edge_attr_chem, chrom_edge_attr_dist
seq_feat                            # [1, 21] sequence vector
kda                                 # [1] molecular weight
y                                   # [1, 2] (brightness, emission)
pdb_code                            # str, e.g. "7ZCT"
```

### `src/fp_gnn/model.py`

This module defines the neural architecture used for prediction, consisting of three main components (classes).

- `ChromMPNN`: NNConv over the chromophore graph, 3 message-passing rounds, GRU update step, returns one H-dim vector per graph via `global_add_pool`. Node embeddings are aggregated into a single graph-level representation using global addition pooling. We set H=64 and did not change it.
- `ProteinMPNN`: same shape, over the residue-contact graph.
- `FPNet`: composes the two MPNNs plus an MLP encoder over the 21-dim `seq_feat`. Concatenates `[prot_emb, chrom_emb, seq_emb, kda_z]` (3H + 1 = 193 dims at H=64). Specifically, it concatenates the protein embedding, chromophore embedding, sequence-level features, and normalized molecular weight into a single vector. For a hidden size of 64, this produces a 193-dimensional input to a two-layer MLP head, which outputs predictions for brightness and emission.

### `src/fp_gnn/lit_module.py`

This module defines the training infrastructure built on PyTorch Lightning, along with deterministic data splitting and normalization of our data. The function `kfold_split(n, k=5, fold=0, n_val=17, seed=0)` implements a reproducible K-fold partitioning scheme in which our dataset N=118 is first shuffled with a fixed seed and divided into five folds, with remainder samples distributed across the initial folds. One fold is held out as test data, while the remaining folds are reshuffled and split into training and validation sets using a fold-specific seed. This ensures that validation sets vary across folds without affecting the independence of the test partition.

The function `compute_zscore_stats(targets, kdas)` computes normalization statistics (train-fold means and stds) for both target variables (brightness and emission wavelenght) and molecular weight (kDa) features using only training data to avoid leakage. We do that because the emission wavelenght is a way bigger number than the brightness (emission scales between 200 and 800, and brightness between 0 and 100). So for our training, the model is trained on z-scores. The `FluorLitModule(net, train_dataset, val_dataset, test_dataset, batch_size=8, lr=1e-3)` class wraps the neural network into a full training loop. During initialization, it computes and stores normalization statistics as buffers, ensuring they move automatically to GPU during training. The training_step computes mean squared error in normalized space and logs epoch-level training loss for visualization purposes (one loss per epoch). The _eval_step method evaluates model performance in both normalized and original units, so that we can report interpretable errors in physical units. Finally, the test_step is done the same way as eval, but we save the predictions to a .csv file (the `train.py` script writes those to `test_predictions.csv` after `trainer.test()` returns, and the `analyze_sweep.py` script reads them for the pred-vs-true scatter.)

## D. OVERVIEW OF THE RESULTS

## Headline

| target           | predict-mean baseline | `FPNet` 5-fold MAE | reduction |
|------------------|----------------------:|-------------------:|----------:|
| emission (nm)    |                 49.90 | **28.42 ± 2.72**   | **−43 %** |
| brightness       |                 22.51 | 21.98 ± 4.13       | ~2 %      |

The naive baselines come from predicting the per-target mean of the full dataset (22.51 / 49.90).

## Per-fold breakdown

| fold | n_test | best epoch | MAE bright | MAE emis (nm) |
|-----:|------:|----------:|-----------:|--------------:|
|    0 |    24 |        25 |      19.64 |        31.57  |
|    1 |    24 |        23 |      16.01 |        25.61  |
|    2 |    24 |        70 |      26.25 |        28.02  |
|    3 |    23 |        34 |      24.55 |        26.04  |
|    4 |    23 |        43 |      23.45 |        30.87  |

Best epoch ranges 23–70 across folds: early stopping plus best-checkpoint selection is what makes this good for each fold, since a fixed cutoff would have cut some folds short and overshot others.

## Reading

- **Emission is genuinely predicted.** ~28 nm error on a 234-nm range is a quarter of the dynamic range, half of the target std, and the per-fold spread is tight (2.7 nm). The pred-vs-true scatter shows a clear y=x trend.
- **Brightness is at the floor.** 22.0 vs naive 22.5 is essentially zero improvement; the scatter shows predictions clamped near the training mean. We interpretate those results in the report, but basically Brightness depends on quantum yield and extinction coefficient, which are set by very local properties of the chromophore environment (rigid packing, H-bond patterns, electric field) that Cα-level structure, mass, and AA composition don't carry.

## Reproducing

Please refer to the README to know how to run the results.

```bash
uv run python scripts/process.py
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python scripts/train.py \
    --fold $fold --exp-name kfold5
done
uv run python scripts/analyze_sweep.py --exp-name kfold5
```
