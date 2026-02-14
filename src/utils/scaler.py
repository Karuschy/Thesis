import torch

class ChannelStandardizer:
    """
    Standardize per-channel: x[c] = (x[c] - mean[c]) / std[c]
    for tensors shaped [N,C,H,W] or [C,H,W].
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> "ChannelStandardizer":
        if x.dim() != 4:
            raise ValueError(f"Expected [N,C,H,W], got {tuple(x.shape)}")
        # mean/std over N,H,W for each channel
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)
        self.std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() before transform().")
        if x.dim() == 3:
            # Unsqueeze to [1,C,H,W], transform, then squeeze back to [C,H,W]
            return ((x.unsqueeze(0) - self.mean) / self.std).squeeze(0)
        if x.dim() == 4:
            return (x - self.mean) / self.std
        raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got {tuple(x.shape)}")

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        if x.dim() == 3:
            # Unsqueeze to [1,C,H,W], inverse, then squeeze back to [C,H,W]
            return (x.unsqueeze(0) * self.std + self.mean).squeeze(0)
        if x.dim() == 4:
            return x * self.std + self.mean
        raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got {tuple(x.shape)}")
