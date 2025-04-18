import torch
from torch import nn

class MonotonicRegularizationLoss(nn.Module):
    def __init__(self, lambda_latency=0.1, lambda_variance=0.1):
        super().__init__()
        self.lambda_latency = lambda_latency
        self.lambda_variance = lambda_variance

    def forward(self, alpha: torch.Tensor):
        latency_loss = self.compute_latency_loss(alpha)
        variance_loss = self.compute_variance_loss(alpha)
        return self.lambda_latency * latency_loss + self.lambda_variance * variance_loss
    
    def compute_variance_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes alignment variance loss (L_variance).
        alpha: (N, H, T, S)
        returns: scalar tensor
        """
        N, H, T, S = alpha.shape
        position_ids = torch.arange(1, S + 1, device=alpha.device).view(1, 1, 1, S)  # (1, 1, 1, S)
        
        expected_pos = (alpha * position_ids).sum(-1)  # (N, H, T)
        expected_pos_sq = (alpha * position_ids**2).sum(-1)  # (N, H, T)

        variance = expected_pos_sq - expected_pos**2  # (N, H, T)

        # Optionally average over heads
        variance = variance.mean(1)  # (N, T)

        return variance.mean()

    def compute_latency_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes expected delay loss (L_latency).
        alpha: (N, H, T, S)
        returns: scalar tensor
        """
        N, H, T, S = alpha.shape
        position_ids = torch.arange(1, S + 1, device=alpha.device).view(1, 1, 1, S)  # (1, 1, 1, S)
        
        expected_delay = (alpha * position_ids).sum(-1)  # (N, H, T)
        
        # Optionally average over heads
        expected_delay = expected_delay.mean(1)  # (N, T)
        
        return expected_delay.mean()