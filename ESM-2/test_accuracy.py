import os
import numpy as np

true_folder = "" # ground truth folder
pred_folder = "" # predicted folder

def load_matrix(path):
    return np.loadtxt(path, dtype=int)

MISMATCH_STRATEGY = "truncate"

TP, FP, FN, TN = 0, 0, 0, 0

pred_map = {}
for filename in os.listdir(pred_folder):
    if filename.endswith(".txt"):
        base = filename.replace(".pdb_pred", "")  # e.g. "1BB3.pdb_pred.txt" -> "1BB3.txt"
        pred_map[base] = filename

for filename in os.listdir(true_folder):
    if filename.endswith(".txt"):
        true_path = os.path.join(true_folder, filename)

        if filename not in pred_map:
            print(f"No prediction found for {filename}")
            continue
        pred_path = os.path.join(pred_folder, pred_map[filename])

        true_mat = load_matrix(true_path)
        pred_mat = load_matrix(pred_path)

        if true_mat.shape != pred_mat.shape:
            if MISMATCH_STRATEGY == "skip":
                print(f"Skipping {filename}, shape mismatch: {true_mat.shape} vs {pred_mat.shape}")
                continue
            elif MISMATCH_STRATEGY == "truncate":
                min_shape = min(true_mat.shape[0], pred_mat.shape[0])
                true_mat = true_mat[:min_shape, :min_shape]
                pred_mat = pred_mat[:min_shape, :min_shape]
                print(f"Truncated {filename} to {true_mat.shape}")
            elif MISMATCH_STRATEGY == "pad":
                max_shape = max(true_mat.shape[0], pred_mat.shape[0])
                padded_true = np.zeros((max_shape, max_shape), dtype=int)
                padded_pred = np.zeros((max_shape, max_shape), dtype=int)
                padded_true[:true_mat.shape[0], :true_mat.shape[1]] = true_mat
                padded_pred[:pred_mat.shape[0], :pred_mat.shape[1]] = pred_mat
                true_mat, pred_mat = padded_true, padded_pred
                print(f"Padded {filename} to {true_mat.shape}")

        # confusion matrix terms
        TP += np.sum((true_mat == 1) & (pred_mat == 1))
        FP += np.sum((true_mat == 0) & (pred_mat == 1))
        FN += np.sum((true_mat == 1) & (pred_mat == 0))
        TN += np.sum((true_mat == 0) & (pred_mat == 0))

# compute metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
