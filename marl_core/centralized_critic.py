# centralized_critic.py
import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    """
    Simple MLP critic that takes global state embedding and outputs V(s).
    Keep it small for CPU training.
    """
    def __init__(self, input_dim: int = 384, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, state_emb: torch.Tensor):
        # state_emb: [batch, input_dim] or [input_dim]
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
        v = self.net(state_emb)
        return v.squeeze(-1)  # [batch] or scalar
