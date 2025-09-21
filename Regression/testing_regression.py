import os
import numpy as np
import joblib

# Letter → number mapping
LETTER_TO_NUM = {chr(i+64): i for i in range(1, 27)}

def predict_contact_map(model, seq):
    """Build features for all residue pairs in seq and predict contact map"""
    N = len(seq)
    X = []
    pairs = []
    for i in range(N):
        for j in range(i+1, N):   # upper triangle only
            aa_i = LETTER_TO_NUM.get(seq[i], 0)
            aa_j = LETTER_TO_NUM.get(seq[j], 0)
            seq_dist = abs(i - j)
            X.append([aa_i, aa_j, seq_dist])
            pairs.append((i, j))
    
    X = np.array(X)
    y_pred = model.predict(X)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Build full NxN contact map
    cmap_pred = np.zeros((N, N), dtype=int)
    for (i, j), val in zip(pairs, y_pred_binary):
        cmap_pred[i, j] = val
        cmap_pred[j, i] = val   # symmetric

    return cmap_pred

if __name__ == "__main__":
    # ---------------------------
    # Paths
    # ---------------------------
    model_path = "linear_model.pkl"   # trained model
    test_folder = "test_npz"     # folder with test npz files

    # Load trained model
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Loop over test files
    for fname in os.listdir(test_folder):
        if not fname.endswith(".npz"):
            continue
        npz_path = os.path.join(test_folder, fname)
        data = np.load(npz_path, allow_pickle=True)
        seq = data["seq"].item() if data["seq"].shape == () else str(data["seq"])
        cmap_true = data["cmap"]

        # Predict contact map
        cmap_pred = predict_contact_map(model, seq)

        print(f"\n{fname}")
        print("Sequence length:", len(seq))
        print("True contact map shape:", cmap_true.shape)
        print("Pred contact map shape:", cmap_pred.shape)

        # Example: show first 10x10 block of predicted vs true
        print("True (10x10):\n", cmap_true[:10, :10])
        print("Pred (10x10):\n", cmap_pred[:10, :10])
