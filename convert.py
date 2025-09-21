import os
import numpy as np
from Bio.PDB import PDBParser

# Mapping from 3-letter to 1-letter amino acid codes
AA_MAP = {
    "ALA":"A","ARG":"R","ASP":"D","CYS":"C","CYX":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","HIE":"H",
    "ILE":"I","LEU":"L","LYS":"K","MET":"M","ASN":"N",
    "PHE":"F","PRO":"P","SEC":"U","SER":"S","THR":"T",
    "TRP":"W","TYR":"Y","VAL":"V"
}

def pdb_to_seq_and_contacts(pdb_file, threshold=8.0):
    """
    Reads a PDB file, extracts sequence and builds contact map
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)

    seq = []
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in AA_MAP and "CA" in residue:
                    seq.append(AA_MAP[resname])
                    coords.append(residue["CA"].get_coord())
    if len(seq) == 0:
        return None, None  # skip empty
    
    seq = "".join(seq)
    coords = np.array(coords)

    # Build contact map
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff**2).sum(-1))
    cmap = (dists < threshold).astype(np.int8)
    np.fill_diagonal(cmap, 0)

    return seq, cmap


def process_pdb_folder(folder, out_folder, limit=100):
    os.makedirs(out_folder, exist_ok=True)
    pdb_files = [f for f in os.listdir(folder) if f.endswith(".pdb")]
    pdb_files = pdb_files[:limit]

    for idx, pdb_name in enumerate(pdb_files):
        pdb_path = os.path.join(folder, pdb_name)
        seq, cmap = pdb_to_seq_and_contacts(pdb_path)

        if seq is None:
            print(f"Skipping {pdb_name} (no residues found)")
            continue

        out_name = os.path.splitext(pdb_name)[0] + ".npz"
        out_path = os.path.join(out_folder, out_name)
        np.savez_compressed(out_path, seq=seq, cmap=cmap)
        print(f"[{idx+1}/{len(pdb_files)}] Saved {out_path} (len={len(seq)})")


# Example usage
if __name__ == "__main__":
    pdb_folder = ""      # folder with pdb files
    out_folder = ""      # output folder for npz
    process_pdb_folder(pdb_folder, out_folder, limit=100)
