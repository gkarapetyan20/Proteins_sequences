import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# ----------------------------
# Amino acid one-hot encoder
# ----------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical amino acids
AA_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse=False)
AA_ENCODER.fit(np.array(AA_LIST).reshape(-1, 1))

def encode_aa(aa):
    """One-hot encode a single amino acid."""
    return AA_ENCODER.transform(np.array([[aa]]))[0]

# ----------------------------
# Predict contact matrix for one sequence
# ----------------------------
def predict_contact_matrix(sequence, model):
    N = len(sequence)
    matrix = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            try:
                aa_i = encode_aa(sequence[i])
                aa_j = encode_aa(sequence[j])
            except Exception:
                # unknown character → skip or fill with zeros
                aa_i = np.zeros(len(AA_LIST))
                aa_j = np.zeros(len(AA_LIST))

            seq_dist = np.array([abs(i - j)])
            features = np.concatenate([aa_i, aa_j, seq_dist]).reshape(1, -1)

            pred = model.predict(features)[0]
            print(pred)
            matrix[i, j] = pred
            matrix[j, i] = pred  # enforce symmetry
    return matrix

# ----------------------------
# Save matrix to txt file
# ----------------------------
def save_matrix_to_txt(matrix, filename):
    np.savetxt(filename, matrix, fmt='%d')
    print(f"Matrix saved to {filename}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model_path = ""  # updated model name
    sequence = ""   #Sequence
    output_file = ""

    # Load model
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Predict matrix
    matrix = predict_contact_matrix(sequence, model)

    # Save to txt
    save_matrix_to_txt(matrix, output_file)
