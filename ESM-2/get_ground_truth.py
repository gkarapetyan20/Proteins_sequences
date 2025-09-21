import os
import numpy as np
from Bio.PDB import PDBParser

# -----------------------------
# Parameters
# -----------------------------
pdb_folder = ""        # folder with PDB files
output_folder = ""  # folder to save ground truth
cutoff = 8.0                 # distance cutoff in angstroms

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize PDB parser
parser = PDBParser(QUIET=True)

# -----------------------------
# Process all PDB files
# -----------------------------
for pdb_file in os.listdir(pdb_folder):
    if not pdb_file.endswith(".pdb"):
        continue

    pdb_path = os.path.join(pdb_folder, pdb_file)
    structure = parser.get_structure(pdb_file, pdb_path)

    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())

    ca_coords = np.array(ca_coords)
    n_residues = len(ca_coords)
    if n_residues == 0:
        print(f"No Cα atoms found in {pdb_file}, skipping.")
        continue

    # Compute contact map (using broadcasting for speed)
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    contact_map = (distances < cutoff).astype(int)

    # Save contact map
    output_txt = os.path.join(output_folder, pdb_file.replace(".pdb", ".txt"))
    np.savetxt(output_txt, contact_map, fmt='%d')
    print(f"Saved contact map for {pdb_file} → {output_txt}")

print("All PDB files processed.")
