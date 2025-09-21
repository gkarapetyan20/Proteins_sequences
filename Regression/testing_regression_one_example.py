import numpy as np
import joblib

# Letter → number mapping
LETTER_TO_NUM = {chr(i+64): i for i in range(1, 27)}

def predict_contact_map(model, seq):
    """Build NxN predicted contact map from one sequence"""
    N = len(seq)
    X = []
    pairs = []

    for i in range(N):
        for j in range(i+1, N):   # only upper triangle
            aa_i = LETTER_TO_NUM.get(seq[i], 0)
            aa_j = LETTER_TO_NUM.get(seq[j], 0)
            seq_dist = abs(i - j)
            X.append([aa_i, aa_j, seq_dist])
            pairs.append((i, j))

    X = np.array(X)
    y_pred = model.predict(X)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Build full symmetric contact map
    cmap_pred = np.zeros((N, N), dtype=int)
    for (i, j), val in zip(pairs, y_pred_binary):
        cmap_pred[i, j] = val
        cmap_pred[j, i] = val

    return cmap_pred

if __name__ == "__main__":
    # Example sequence
    seq = "SCP"

    # Load trained model
    model = joblib.load("linear_model.pkl")
    print("Model loaded")

    # Predict matrix
    cmap_pred = predict_contact_map(model, seq)

    print("Sequence length:", len(seq))
    print("Predicted contact map shape:", cmap_pred.shape)

    # Show first 10x10 block
    print("Pred (10x10):\n", cmap_pred[:100, :10])
