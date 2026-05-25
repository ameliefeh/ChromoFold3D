from pathlib import Path

import numpy as np

CHROMOPHORE_CODES = {
    "NRQ",
    "CRQ",
    "NRP",
    "CH6",
    "CRO",
    "5SQ",
    "4M9",
    "CR2",
    "OFM",
    "CR8",
    "CFY",
    "OIM",
    "CH7",
    "GYS",
    "WCR",
    "GYC",
    "DYG",
    "FAD",
    "PIA",
    "CCY",
    "BLR",
    "CRF",
    "NYG",
    "CR7",
    "FMN",
    "B2H",
    "SWG",
    "CSH",
    "BJF",
}


def get_chromophore_code(pdb_path: str | Path) -> str:
    """Return the first HETATM residue name in `pdb_path` that matches
    a known chromophore code, or raise ValueError if none is found."""
    pdb_path = Path(pdb_path)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            residue_name = line[17:20].strip()
            if residue_name in CHROMOPHORE_CODES:
                return residue_name
    raise ValueError(f"No known chromophore found in {pdb_path}")


def get_chromophore_pdb_block(pdb_path: str | Path, residue_code: str) -> str:
    """Return all HETATM lines for `residue_code` in chain A as a PDB
    block string (terminated by 'END\\n'). Raises ValueError if chain A
    is absent.

    If multiple residue copies exist in chain A (e.g. multiple NRQ
    instances), the lowest resseq is used and a warning is printed.
    """
    pdb_path = Path(pdb_path)

    chain_a_resseqs: set[str] = set()
    matched_lines_per_resseq: dict[str, list[str]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != residue_code:
                continue
            if line[21] != "A":
                continue
            resseq = line[22:26].strip()
            chain_a_resseqs.add(resseq)
            matched_lines_per_resseq.setdefault(resseq, []).append(line)

    if not chain_a_resseqs:
        raise ValueError(f"Residue '{residue_code}' not found in chain A of {pdb_path.name}")

    if len(chain_a_resseqs) > 1:
        print(
            f"[WARN] Multiple '{residue_code}' copies in chain A "
            f"({sorted(chain_a_resseqs)}); using lowest resseq."
        )

    chosen = sorted(chain_a_resseqs, key=lambda s: int(s))[0]
    return "".join(matched_lines_per_resseq[chosen]) + "END\n"


def get_protein_residue_ca(pdb_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Parse ATOM records from `pdb_path`, keep chain A and altloc ' '
    or 'A', skip waters (HOH), and return one Cα atom per residue.

    Returns:
        residue_names: list[str], length N
        ca_coords:    np.ndarray of shape (N, 3); x, y, z in Å
    """
    pdb_path = Path(pdb_path)

    # Order-preserving collection: keyed by (chain, resseq, icode)
    seen_residues = []
    ca_per_residue = {}
    name_per_residue = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain = line[21]
            if chain != "A":
                continue
            res_name = line[17:20].strip()
            if res_name == "HOH":
                continue
            atom_name = line[12:16].strip()
            resseq = line[22:26].strip()
            icode = line[26]
            res_id = (chain, resseq, icode)

            if res_id not in name_per_residue:
                seen_residues.append(res_id)
                name_per_residue[res_id] = res_name

            if atom_name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_per_residue[res_id] = (x, y, z)

    residue_names = []
    coords = []
    for res_id in seen_residues:
        if res_id not in ca_per_residue:
            print(f"[WARN] residue {res_id} in {pdb_path.name} has no Cα atom; skipping")
            continue
        residue_names.append(name_per_residue[res_id])
        coords.append(ca_per_residue[res_id])

    return residue_names, np.asarray(coords, dtype=np.float32)
