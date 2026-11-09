import torch
from prepare import prepare_data
from model import ECGNet, cross_entropy

def run_grad_check():
    data = prepare_data()
    X_train_t = data["X_train"]
    y_train_t = data["y_train"]

    torch.manual_seed(0)
    model = ECGNet()

    B = 64
    x_check = X_train_t[:B]
    y_check = y_train_t[:B]

    logits_check, a2_check = model(x_check, return_activations=True)
    loss_check = cross_entropy(logits_check, y_check)
    loss_check.backward()

    # Hand-computed output layer gradients
    y_hat = torch.softmax(logits_check, dim=1)
    y_one_hot = torch.eye(5)[y_check]

    dJ_dW3_hand = torch.matmul((y_hat - y_one_hot).t(), a2_check) / B
    dJ_db3_hand = torch.sum(y_hat - y_one_hot, dim=0) / B

    diff_W3 = (dJ_dW3_hand - model.W3.grad).abs().max().item()
    diff_b3 = (dJ_db3_hand - model.b3.grad).abs().max().item()

    print(f"Max absolute difference for W^[3]: {diff_W3:.4e}")
    print(f"Max absolute difference for b^[3]: {diff_b3:.4e}")
    return diff_W3, diff_b3

if __name__ == "__main__":
    run_grad_check()
