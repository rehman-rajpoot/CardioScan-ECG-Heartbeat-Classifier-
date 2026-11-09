import numpy as np
import pandas as pd
import torch
from load_data import load_raw

def prepare_data(data_dir="data"):
    raw = load_raw(data_dir)
    train_df = raw["train"]
    test_df = raw["test"]

    # 1.2 Separate signal (0..186) and label (187)
    X_train_np = train_df.iloc[:, :187].values.astype(np.float32)
    y_train_np = train_df.iloc[:, 187].values.astype(np.int64)

    X_test_np = test_df.iloc[:, :187].values.astype(np.float32)
    y_test_np = test_df.iloc[:, 187].values.astype(np.int64)

    # 1.6 Split training file into train and validation with exact given code
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = X_train_np[train_idx]
    y_train = y_train_np[train_idx]
    X_val = X_train_np[val_idx]
    y_val = y_train_np[val_idx]

    # 1.7 Convert to torch.float32 and torch.long tensors
    X_train_t = torch.from_numpy(X_train).to(dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(dtype=torch.long)
    X_val_t = torch.from_numpy(X_val).to(dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val).to(dtype=torch.long)
    X_test_t = torch.from_numpy(X_test_np).to(dtype=torch.float32)
    y_test_t = torch.from_numpy(y_test_np).to(dtype=torch.long)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "y_train_raw": y_train_np,
        "y_test_raw": y_test_np,
        "sig_min": float(X_train_np.min()),
        "sig_max": float(X_train_np.max())
    }

if __name__ == "__main__":
    data = prepare_data()
    print("Train shape:", data["train_df"].shape)
    print("Test shape :", data["test_df"].shape)
    print(f"Signal min : {data['sig_min']:.4f}, max: {data['sig_max']:.4f}")

    print("\nRows per class (Train):")
    print(pd.Series(data["y_train_raw"]).value_counts().sort_index())
    print("\nRows per class (Test):")
    print(pd.Series(data["y_test_raw"]).value_counts().sort_index())

    print("\nSplit row counts:")
    print("Train rows     :", len(data["X_train"]))
    print("Validation rows:", len(data["X_val"]))
    print("Test rows      :", len(data["X_test"]))
