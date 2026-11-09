from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from prepare import prepare_data
from model import ECGNet
from metrics import accuracy, confusion_matrix, per_class_accuracy, balanced_accuracy

def run_test(model_path="results/best_model.pt"):
    data = prepare_data()
    X_test_t = data["X_test"]
    y_test_t = data["y_test"]

    model = ECGNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds_t = torch.argmax(test_logits, dim=1)

    y_test_pred = test_preds_t.numpy()
    y_test_true = y_test_t.numpy()

    test_acc = accuracy(y_test_true, y_test_pred)
    C_test = confusion_matrix(y_test_true, y_test_pred, num_classes=5)
    per_class_acc = per_class_accuracy(C_test)
    bal_acc = balanced_accuracy(C_test)
    always_N_acc = float(np.sum(y_test_true == 0) / len(y_test_true))

    print(f"Test Accuracy      : {test_acc*100:.2f}%")
    print(f"Balanced Accuracy  : {bal_acc*100:.2f}%")
    print(f"Always-'N' Accuracy: {always_N_acc*100:.2f}%")

    short_names = ["N", "S", "V", "F", "Q"]
    print("\n--- Per-Class Accuracy ---")
    for k, name in enumerate(short_names):
        print(f"Class {name}: {per_class_acc[k]*100:.2f}%")

    print("\n--- 5x5 Confusion Matrix ---")
    cm_df = pd.DataFrame(C_test,
                         index=[f"True {k} ({short_names[k]})" for k in range(5)],
                         columns=[f"Pred {k} ({short_names[k]})" for k in range(5)])
    print(cm_df.to_string())

    # Save heatmap
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6.5))
    plt.imshow(C_test, cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(5)
    plt.xticks(ticks, short_names)
    plt.yticks(ticks, short_names)

    thresh = C_test.max() / 2.0
    for i in range(5):
        for j in range(5):
            color = "white" if C_test[i, j] > thresh else "black"
            plt.text(j, i, f"{C_test[i, j]:,}", horizontalalignment="center", color=color)

    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    print(f"\nSaved confusion matrix heatmap to {results_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    run_test()
