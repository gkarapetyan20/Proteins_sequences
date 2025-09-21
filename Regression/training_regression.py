import os
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Letter → number mapping
LETTER_TO_NUM = {chr(i+64): i for i in range(1, 27)}

def npz_to_pairs(npz_file):
    """Extract features and labels from one .npz file"""
    data = np.load(npz_file, allow_pickle=True)
    seq = data["seq"].item() if data["seq"].shape == () else str(data["seq"])
    cmap = data["cmap"]
    
    N = len(seq)
    X = []
    y = []
    for i in range(N):
        for j in range(i+1, N):   # skip diagonal
            aa_i = LETTER_TO_NUM.get(seq[i], 0)
            aa_j = LETTER_TO_NUM.get(seq[j], 0)
            seq_dist = abs(i - j)
            X.append([aa_i, aa_j, seq_dist])
            y.append(int(cmap[i, j]))
    return np.array(X), np.array(y)

def load_dataset_from_folder(folder, limit=None):
    """Load all npz files and build full dataset"""
    X_all, y_all = [], []
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if limit:
        files = files[:limit]  # optional for debugging
    for idx, fname in enumerate(files):
        npz_path = os.path.join(folder, fname)
        try:
            X, y = npz_to_pairs(npz_path)
            X_all.append(X)
            y_all.append(y)
            print(f"[{idx+1}/{len(files)}] Loaded {fname} (pairs={len(y)})")
        except Exception as e:
            print(f"Skipped {fname}: {e}")
    return np.vstack(X_all), np.concatenate(y_all)

# ----------------------------
# Train and Save
# ----------------------------
npz_folder = "/home/aimaster/protain/npz"   # your folder with npz files

X, y = load_dataset_from_folder(npz_folder, limit=None)
print("Total dataset size:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model to disk
joblib.dump(model, "linear_model.pkl")
print("Model saved to linear_model.pkl")

# ----------------------------
# Load and Test
# ----------------------------
loaded_model = joblib.load("linear_model.pkl")
print("Model loaded")

y_pred = loaded_model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_binary)
print("Test Accuracy:", acc)

# Optional: inspect weights
print("Model coefficients:", loaded_model.coef_)
print("Model intercept:", loaded_model.intercept_)
