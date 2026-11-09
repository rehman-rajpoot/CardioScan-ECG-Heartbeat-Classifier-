import numpy as np
import torch
import torch.nn as nn

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Architecture: 187 -> 128 -> 64 -> 5
        n0, n1, n2, n3 = 187, 128, 64, 5

        # He initialisation: W ~ N(0, sqrt(2 / n_prev)), b = 0
        std1 = np.sqrt(2.0 / n0)
        self.W1 = nn.Parameter(torch.empty(n1, n0).normal_(mean=0.0, std=std1))
        self.b1 = nn.Parameter(torch.zeros(n1))

        std2 = np.sqrt(2.0 / n1)
        self.W2 = nn.Parameter(torch.empty(n2, n1).normal_(mean=0.0, std=std2))
        self.b2 = nn.Parameter(torch.zeros(n2))

        std3 = np.sqrt(2.0 / n2)
        self.W3 = nn.Parameter(torch.empty(n3, n2).normal_(mean=0.0, std=std3))
        self.b3 = nn.Parameter(torch.zeros(n3))

    def relu(self, z):
        return torch.clamp(z, min=0.0)

    def forward(self, x, return_activations=False):
        z1 = torch.matmul(x, self.W1.t()) + self.b1
        a1 = self.relu(z1)

        z2 = torch.matmul(a1, self.W2.t()) + self.b2
        a2 = self.relu(z2)

        z3 = torch.matmul(a2, self.W3.t()) + self.b3

        if return_activations:
            return z3, a2
        return z3

def cross_entropy(logits, y):
    log_sum_exp = torch.logsumexp(logits, dim=1)
    true_logits = logits[range(len(y)), y]
    loss = -true_logits + log_sum_exp
    return loss.mean()

if __name__ == "__main__":
    torch.manual_seed(0)
    model = ECGNet()
    # Layer 1 parameters (n0 -> n1)
    W1_params = n1 * n0
    b1_params = n1
    L1_params = W1_params + b1_params

    # Layer 2 parameters (n1 -> n2)
    W2_params = n2 * n1
    b2_params = n2
    L2_params = W2_params + b2_params

    # Layer 3 parameters (n2 -> n3)
    W3_params = n3 * n2
    b3_params = n3
    L3_params = W3_params + b3_params

    # Total trainable parameters
    total_trainable_parameters = L1_params + L2_params + L3_params

    print(f"Total trainable parameters: {total_trainable_parameters}")
    print(f"Layer 1 trainable parameters: {L1_params}")
    print(f"Layer 2 trainable parameters: {L2_params}")
    print(f"Layer 3 trainable parameters: {L3_params}")

