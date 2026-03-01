import torch

class ChannelStandardizer:
    """
    Standardize per-channel: x[c] = (x[c] - mean[c]) / std[c]
    for tensors shaped [N,C,H,W] or [C,H,W].

    If ``log_transform=True`` the forward path applies ``log(clamp(x))``
    before z-scoring, and the inverse path applies ``exp`` after un-z-scoring,
    so the model always works in log-IV space while callers see raw IV.
    """
    LOG_CLAMP_MIN = 1e-4  # floor before log to avoid -inf

    def __init__(self, eps: float = 1e-6, log_transform: bool = False):
        self.eps = eps
        self.log_transform = log_transform
        self.mean = None
        self.std = None

    def to(self, device: torch.device | str) -> "ChannelStandardizer":
        """Move scaler tensors to the given device (for GPU penalty calc)."""
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    # ── helpers ──────────────────────────────────────────────────────────
    def _apply_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp(min=self.LOG_CLAMP_MIN))

    def _apply_exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    # ── core API ─────────────────────────────────────────────────────────
    def fit(self, x: torch.Tensor) -> "ChannelStandardizer":
        if x.dim() != 4:
            raise ValueError(f"Expected [N,C,H,W], got {tuple(x.shape)}")
        if self.log_transform:
            x = self._apply_log(x)
        # mean/std over N,H,W for each channel
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)
        self.std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() before transform().")
        if self.log_transform:
            x = self._apply_log(x)
        if x.dim() == 3:
            return ((x.unsqueeze(0) - self.mean) / self.std).squeeze(0)
        if x.dim() == 4:
            return (x - self.mean) / self.std
        raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got {tuple(x.shape)}")

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        if x.dim() == 3:
            out = (x.unsqueeze(0) * self.std + self.mean).squeeze(0)
        elif x.dim() == 4:
            out = x * self.std + self.mean
        else:
            raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got {tuple(x.shape)}")
        if self.log_transform:
            out = self._apply_exp(out)
        return out
