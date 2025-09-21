import os
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)

# ----------------------------
# Amino acid encoder (must match training!)
# ----------------------------
from sklearn.preprocessing import OneHotEncoder
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical amino acids
AA_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse=False)
AA_ENCODER.fit(np.array(AA_LIST).reshape(-1, 1))

def encode_aa(aa):
    """One-hot encode a single amino acid"""
    arr = np.array([[aa]])
    return AA_ENCODER.transform(arr)[0]

# ----------------------------
# Extract features/labels from one .npz
# ----------------------------
def npz_to_pairs(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    seq = data["seq"].item() if data["seq"].shape == () else str(data["seq"])
    cmap = data["cmap"]

    N = len(seq)
    X, y = [], []
    for i in range(N):
        for j in range(i + 1, N):  # skip diagonal
            aa_i = encode_aa(seq[i])
            aa_j = encode_aa(seq[j])
            seq_dist = np.array([abs(i - j)])
            features = np.concatenate([aa_i, aa_j, seq_dist])
            X.append(features)
            y.append(int(cmap[i, j]))
    return np.array(X), np.array(y)

# ----------------------------
# Evaluate model on all files in a folder
# ----------------------------
def evaluate_folder(model_path, npz_folder):
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    files = [f for f in os.listdir(npz_folder) if f.endswith(".npz")]
    if not files:
        print("No .npz files found in folder")
        return

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for idx, fname in enumerate(files):
        npz_path = os.path.join(npz_folder, fname)
        try:
            X, y = npz_to_pairs(npz_path)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)

            print(f"[{idx+1}/{len(files)}] {fname} → "
                  f"Acc: {acc:.4f}, Prec: {precision:.4f}, "
                  f"Rec: {recall:.4f}, F1: {f1:.4f}")

            all_y_true.append(y)
            all_y_pred.append(y_pred)
            if y_prob is not None:
                all_y_prob.append(y_prob)

        except Exception as e:
            print(f"Skipped {fname}: {e}")

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_prob = np.concatenate(all_y_prob) if all_y_prob else None

    print("\nOverall metrics:")
    print(f"Accuracy : {accuracy_score(all_y_true, all_y_pred):.4f}")
    print(f"Precision: {precision_score(all_y_true, all_y_pred):.4f}")
    print(f"Recall   : {recall_score(all_y_true, all_y_pred):.4f}")
    print(f"F1-score : {f1_score(all_y_true, all_y_pred):.4f}")

    print("\nClassification report:")
    print(classification_report(all_y_true, all_y_pred, digits=4))

    if all_y_prob is not None:
        try:
            auc = roc_auc_score(all_y_true, all_y_prob)
            print(f"ROC-AUC  : {auc:.4f}")
        except Exception as e:
            print(f" Could not compute ROC-AUC: {e}")

# ----------------------------
# Run evaluation
# ----------------------------
model_path = ""   # updated model name
test_folder = ""

evaluate_folder(model_path, test_folder)
