import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# ----------------------------
# Amino acid alphabet
# ----------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical amino acids
AA_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse=False)
AA_ENCODER.fit(np.array(AA_LIST).reshape(-1, 1))

def encode_aa(aa):
    arr = np.array([[aa]])
    return AA_ENCODER.transform(arr)[0]

# ----------------------------
# Convert one .npz file to features/labels
# ----------------------------
def npz_to_pairs(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    seq = data["seq"].item() if data["seq"].shape == () else str(data["seq"])
    cmap = data["cmap"]

    N = len(seq)
    X, y = [], []
    for i in range(N):
        for j in range(i + 1, N):
            aa_i = encode_aa(seq[i])
            aa_j = encode_aa(seq[j])
            seq_dist = np.array([abs(i - j)])
            features = np.concatenate([aa_i, aa_j, seq_dist])
            X.append(features)
            y.append(int(cmap[i, j]))
    return np.array(X), np.array(y)

def load_dataset_from_folder(folder, limit=None):
    X_all, y_all = [], []
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if limit:
        files = files[:limit]
    for idx, fname in enumerate(files):
        try:
            X, y = npz_to_pairs(os.path.join(folder, fname))
            X_all.append(X)
            y_all.append(y)
        except Exception as e:
            print(f"Skipped {fname}: {e}")
    if not X_all:
        raise RuntimeError("No valid npz files found!")
    return np.vstack(X_all), np.concatenate(y_all)

# ----------------------------
# Training with simulated epochs
# ----------------------------
npz_folder = ""
X, y = load_dataset_from_folder(npz_folder)
print("Total dataset size:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simulate 5 "epochs" by gradually increasing number of trees
trees_per_epoch = 50
epochs = 5

for epoch in range(1, epochs + 1):
    model = RandomForestClassifier(
        n_estimators=trees_per_epoch * epoch,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Epoch {epoch}/{epochs} - Trees: {trees_per_epoch*epoch}")
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}\n")

# Save final model
joblib.dump(model, "contact_rf_model.pkl")
print("Final model saved to contact_rf_model.pkl")
