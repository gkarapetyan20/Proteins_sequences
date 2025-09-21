import os
import torch
import esm
import numpy as np
from Bio.PDB import PDBParser, PPBuilder

# ----------------------------
# 1. Load ESM2 model
# ----------------------------
print("Loading ESM2 model...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# ----------------------------
# 2. Extract sequence from PDB
# ----------------------------
def get_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += str(pp.get_sequence())
    return sequence

# ----------------------------
# 3. Process all PDBs in folder
# ----------------------------
pdb_folder = ""  # change folder path
out_folder = ""  # save as txt
os.makedirs(out_folder, exist_ok=True)

pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
print(f"Found {len(pdb_files)} PDB files.")

for pdb_file in pdb_files:
    pdb_path = os.path.join(pdb_folder, pdb_file)
    name = os.path.splitext(pdb_file)[0]

    try:
        # Extract sequence
        sequence = get_sequence_from_pdb(pdb_path)
        if len(sequence) == 0:
            print(f"Skipping {pdb_file}, no sequence found.")
            continue
        print(f"Processing {pdb_file} ({len(sequence)} aa)")

        # Prepare input
        data = [(name, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Run ESM - only contact maps
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[], return_contacts=True)

        # Process contact map
        for idx, (_, seq), tokens_len, attention_contacts in zip(
            range(len(data)), data, batch_lens, results["contacts"]
        ):
            contact_map = attention_contacts[:tokens_len, :tokens_len].cpu().numpy()

            # Threshold to binary and downcast
            threshold = 0.1
            binary_contact_map = (contact_map > threshold).astype(np.uint8)

            # Save as TXT
            txt_file = os.path.join(out_folder, f"{name}.txt")
            np.savetxt(txt_file, binary_contact_map, fmt="%d")

            print(f"Saved binary contact map for {name} ({binary_contact_map.shape[0]}x{binary_contact_map.shape[1]})")

        # Free memory
        del results, contact_map, binary_contact_map
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")

