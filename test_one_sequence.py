import torch
import torch.nn as nn
import esm
import numpy as np

# ----------------------------
# Load ESM2 model
# ----------------------------
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
device = "cuda" if torch.cuda.is_available() else "cpu"
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# ----------------------------
# Trained NN definition
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

# ----------------------------
# Load trained model weights
# ----------------------------
input_dim = 640  # 2*320 for esm2_t6_8M
predictor_model = ContactPredictor(input_dim).to(device)
predictor_model.load_state_dict(torch.load("ESM-2/contact_predictor_nn.pth", map_location=device))
predictor_model.eval()
print("Loaded trained NN weights")

# ----------------------------
# Contact map prediction
# ----------------------------
def predict_contact_map(residue_embeddings, model, batch_size=20000):
    L, D = residue_embeddings.shape
    preds = np.zeros((L, L), dtype=np.int32)

    for i in range(L):
        X_pairs, idx_pairs = [], []
        for j in range(i + 1, L):
            feat = np.concatenate([residue_embeddings[i], residue_embeddings[j]])
            X_pairs.append(feat)
            idx_pairs.append((i, j))

            if len(X_pairs) >= batch_size:
                X_pairs_tensor = torch.tensor(X_pairs, dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = model(X_pairs_tensor).cpu().numpy()
                for (pi, pj), val in zip(idx_pairs, out):
                    preds[pi, pj] = preds[pj, pi] = 1 if val > 0.5 else 0
                X_pairs, idx_pairs = [], []

        if X_pairs:
            X_pairs_tensor = torch.tensor(X_pairs, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(X_pairs_tensor).cpu().numpy()
            for (pi, pj), val in zip(idx_pairs, out):
                preds[pi, pj] = preds[pj, pi] = 1 if val > 0.5 else 0

    return preds

# ----------------------------
# Example run
# ----------------------------
sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAY"  # <-- example sequence
data_batch = [("example", sequence)]
_, _, batch_tokens = batch_converter(data_batch)
batch_tokens = batch_tokens.to(device)

with torch.no_grad():
    results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
token_reps = results["representations"][6]
residue_embeddings = token_reps[0, 1:len(sequence)+1].cpu().numpy()

# Predict
contact_map_pred = predict_contact_map(residue_embeddings, predictor_model, batch_size=20000)
print(contact_map_pred)
# Save or print result
np.savetxt("example_contact_map.txt", contact_map_pred, fmt="%d")
print("Predicted contact map saved as example_contact_map.txt")
