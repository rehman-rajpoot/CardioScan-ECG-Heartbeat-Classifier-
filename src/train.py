import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from prepare import prepare_data
from model import ECGNet, cross_entropy

def compute_accuracy(model, x, y, batch_size=1024):
    model.eval()
    correct = 0
    total = len(y)
    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb = x[i : i + batch_size]
            yb = y[i : i + batch_size]
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
    return correct / total

def compute_loss(model, x, y, batch_size=1024):
    model.eval()
    total_loss = 0.0
    total = len(y)
    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb = x[i : i + batch_size]
            yb = y[i : i + batch_size]
            logits = model(xb)
            loss = cross_entropy(logits, yb)
            total_loss += loss.item() * len(yb)
    return total_loss / total

def train(alpha, X_train_t, y_train_t, X_val_t, y_val_t, epochs=30, batch_size=128, beta=0.9, weight_decay=1e-4):
    torch.manual_seed(0)
    net = ECGNet()

    velocities = {}
    for name, param in net.named_parameters():
        velocities[name] = torch.zeros_like(param.data)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_state_dict": None
    }

    N_train = len(X_train_t)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(N_train)

        for i in range(0, N_train, batch_size):
            batch_indices = perm[i : i + batch_size]
            xb = X_train_t[batch_indices]
            yb = y_train_t[batch_indices]

            logits = net(xb)
            loss = cross_entropy(logits, yb)
            loss.backward()

            with torch.no_grad():
                for name, param in net.named_parameters():
                    grad = param.grad
                    if "W" in name:
                        g = grad + weight_decay * param.data
                    else:
                        g = grad

                    velocities[name] = beta * velocities[name] + (1.0 - beta) * g
                    param.data = param.data - alpha * velocities[name]
                    param.grad.zero_()

        tr_loss = compute_loss(net, X_train_t, y_train_t)
        va_loss = compute_loss(net, X_val_t, y_val_t)
        tr_acc = compute_accuracy(net, X_train_t, y_train_t)
        va_acc = compute_accuracy(net, X_val_t, y_val_t)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_loss < history["best_val_loss"]:
            history["best_val_loss"] = va_loss
            history["best_epoch"] = epoch
            saved_weights = {}
            for k, v in net.state_dict().items():
                saved_weights[k] = v.clone()
            history["best_state_dict"] = saved_weights

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
                  f"Train Acc: {tr_acc*100:.2f}% | Val Acc: {va_acc*100:.2f}%")

    history["elapsed_time"] = time.time() - t_start
    return history, net

if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data()
    X_train_t, y_train_t = data["X_train"], data["y_train"]
    X_val_t, y_val_t = data["X_val"], data["y_val"]

    learning_rates = [0.01, 0.1, 0.5]
    lr_histories = {}
    lr_summary = []

    for alpha in learning_rates:
        print(f"\n==================== Training with alpha = {alpha} ====================")
        hist, _ = train(alpha, X_train_t, y_train_t, X_val_t, y_val_t)
        lr_histories[alpha] = hist

        best_ep = hist["best_epoch"]
        best_vl = hist["best_val_loss"]
        val_acc_at_best = hist["val_acc"][best_ep - 1]
        lr_summary.append({
            "alpha": alpha,
            "Lowest validation loss": round(best_vl, 4),
            "Validation accuracy at that epoch": f"{val_acc_at_best*100:.2f}%",
            "Best epoch": best_ep
        })

    print("\n--- Table 3: Learning-rate choice ---")
    print(pd.DataFrame(lr_summary).to_string(index=False))

    # Plot validation loss comparison
    epochs_range = range(1, 31)
    plt.figure(figsize=(9, 5))
    colors = {0.01: "blue", 0.1: "orange", 0.5: "green"}
    for alpha in learning_rates:
        plt.plot(epochs_range, lr_histories[alpha]["val_loss"], label=f"alpha = {alpha}", color=colors[alpha], lw=2)
    plt.title("Validation Loss vs. Epoch for Learning Rates {0.01, 0.1, 0.5}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "lr_comparison_val_loss.png", dpi=150)

    best_alpha = 0.01
    lowest_val_loss = lr_histories[0.01]["best_val_loss"]
    for alpha in learning_rates:
        if lr_histories[alpha]["best_val_loss"] < lowest_val_loss:
            lowest_val_loss = lr_histories[alpha]["best_val_loss"]
            best_alpha = alpha

    chosen_hist = lr_histories[best_alpha]
    best_epoch = chosen_hist["best_epoch"]

    # Loss curves for best alpha
    plt.figure(figsize=(9, 4.5))
    plt.plot(epochs_range, chosen_hist["train_loss"], label="Training Loss", color="blue", lw=2)
    plt.plot(epochs_range, chosen_hist["val_loss"], label="Validation Loss", color="red", lw=2)
    plt.axvline(best_epoch, color="black", linestyle=":", label=f"Best Epoch ({best_epoch})")
    plt.title(f"Training and Validation Loss (alpha = {best_alpha})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curves.png", dpi=150)

    # Accuracy curves for best alpha
    plt.figure(figsize=(9, 4.5))
    plt.plot(epochs_range, np.array(chosen_hist["train_acc"]) * 100, label="Training Accuracy", color="blue", lw=2)
    plt.plot(epochs_range, np.array(chosen_hist["val_acc"]) * 100, label="Validation Accuracy", color="green", lw=2)
    plt.axvline(best_epoch, color="black", linestyle=":", label=f"Best Epoch ({best_epoch})")
    plt.title(f"Training and Validation Accuracy (alpha = {best_alpha})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_curves.png", dpi=150)

    torch.save(chosen_hist["best_state_dict"], results_dir / "best_model.pt")
    print(f"Saved best model to {results_dir / 'best_model.pt'}")
