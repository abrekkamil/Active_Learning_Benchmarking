import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_budget_options):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.image_head = nn.Linear(hidden_dim, 1)
        self.budget_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, global_state=None):
        h = self.encoder(x)
        image_logits = self.image_head(h).squeeze(-1)  # [N]

        budget_logit = None
        if global_state is not None:
            g = self.encoder(global_state.unsqueeze(0))          # [1, H]
            budget_logit = self.budget_head(g).squeeze()         # scalar

        return image_logits, budget_logit