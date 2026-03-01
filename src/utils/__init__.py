"""Utility modules for VAE training and evaluation."""

from src.utils.scaler import ChannelStandardizer
from src.utils.training import TrainStats, evaluate, train_epoch, fit_vae
from src.utils.arbitrage import (
    calendar_penalty,
    butterfly_penalty,
    compute_arb_penalty,
    count_calendar_violations,
    count_butterfly_violations,
)
from src.utils.eval import (
    EvalMetrics,
    DetailedEvalResult,
    evaluate_vae,
    compute_per_cell_error,
    get_worst_reconstructions,
    get_best_reconstructions,
)
from src.utils.masking import (
    MaskedEvalMetrics,
    create_random_mask,
    create_structured_mask,
    evaluate_with_masking,
    evaluate_completion_sweep,
    print_completion_summary,
)

__all__ = [
    # Scaling
    "ChannelStandardizer",
    # Training
    "TrainStats",
    "evaluate",
    "train_epoch",
    "fit_vae",
    # Arbitrage
    "calendar_penalty",
    "butterfly_penalty",
    "compute_arb_penalty",
    "count_calendar_violations",
    "count_butterfly_violations",
    # Evaluation
    "EvalMetrics",
    "DetailedEvalResult",
    "evaluate_vae",
    "compute_per_cell_error",
    "get_worst_reconstructions",
    "get_best_reconstructions",
    # Masking
    "MaskedEvalMetrics",
    "create_random_mask",
    "create_structured_mask",
    "evaluate_with_masking",
    "evaluate_completion_sweep",
    "print_completion_summary",
]
