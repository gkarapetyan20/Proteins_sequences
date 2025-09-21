import os
import torch
import torch.nn as nn
import torch.optim as optim
import esm
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ----------------------------
# Load ESM2 model
# ----------------------------
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
device = "cuda" if torch.cuda.is_available() else "cpu"
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# ----------------------------
# Helper functions
# ----------------------------
def extract_sequence_and_ca(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)
    ppb = PPBuilder()
    chain_data = {}
    for model_ in structure:
        for chain in model_:
            seqs, ca_coords = [], []
            for pp in ppb.build_peptides(chain):
                seqs.append(str(pp.get_sequence()))
                for residue in pp:
                    if "CA" in residue:
                        ca_coords.append(residue["CA"].coord)
            if seqs and ca_coords:
                chain_data[chain.id] = {
                    "seq": "".join(seqs),
                    "ca_coords": np.array(ca_coords)
                }
    return chain_data

def compute_contact_map(ca_coords, threshold=8.0):
    dist_matrix = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :],
        axis=-1
    )
    contact_map = (dist_matrix < threshold).astype(np.int8)
    np.fill_diagonal(contact_map, 0)
    return contact_map

# ----------------------------
# PairDataset (lazy pairs)
# ----------------------------
class PairDataset(Dataset):
    def __init__(self, residue_embeddings, contact_map):
        self.embeddings = residue_embeddings
        self.contact_map = contact_map
        self.L = residue_embeddings.shape[0]
        self.indices = [(i, j) for i in range(self.L) for j in range(self.L) if i != j]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        feat = np.concatenate([self.embeddings[i], self.embeddings[j]])
        label = self.contact_map[i, j]
        return torch.tensor(feat, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

# ----------------------------
# Build dataset from folder
# ----------------------------
pdb_folder = "" # pdb folder path 
datasets = []

for pdb_file in os.listdir(pdb_folder):
    if not pdb_file.endswith(".pdb"):
        continue

    pdb_path = os.path.join(pdb_folder, pdb_file)
    chains = extract_sequence_and_ca(pdb_path)

    for chain_id, data in chains.items():
        seq, ca_coords = data["seq"], data["ca_coords"]
        if len(seq) != len(ca_coords) or len(seq) == 0:
            continue

        # ESM embeddings
        data_batch = [(f"{pdb_file}_{chain_id}", seq)]
        _, _, batch_tokens = batch_converter(data_batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_reps = results["representations"][6]
            residue_embeddings = token_reps[0, 1:len(seq)+1].cpu().numpy()

        # Contact map
        contact_map = compute_contact_map(ca_coords)

        # Add dataset for this chain
        datasets.append(PairDataset(residue_embeddings, contact_map))
        print(f"Loaded {pdb_file} chain {chain_id} | seq_len={len(seq)}")

# Merge all chain datasets
full_dataset = ConcatDataset(datasets)
loader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=2)

# ----------------------------
# Define model
# ----------------------------
class ContactPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

input_dim = datasets[0].embeddings.shape[1] * 2
model = ContactPredictor(input_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# Training loop
# ----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(full_dataset):.4f}")

# ----------------------------
# Save model
# ----------------------------
torch.save(model.state_dict(), "contact_predictor_nn.pth")
print("Saved trained NN weights to contact_predictor_nn.pth")
