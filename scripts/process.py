"""
One-shot data processing entry point.

This script builds protein/chromophore graphs for each entry in `data/labels.csv`
and saves the result to `data/processed/data.pt` as a cache.

On later runs, training will load this cached file instead of rebuilding the graphs.

The script should be run again whenever we change `data/labels.csv` or update files in `data/raw/*.pdb` (add more proteins).
"""

from fp_gnn.dataset import FluorProteinDataset

DATA_ROOT = "data"


def main():
    ds = FluorProteinDataset(root=DATA_ROOT)
    print(f"processed {len(ds)} proteins -> {DATA_ROOT}/processed/data.pt")


if __name__ == "__main__":
    main()
