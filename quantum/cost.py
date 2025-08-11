from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class CostWeights:
    lambda_amp: float = 2e-5
    lambda_leak: float = 5e-4
    lambda_xt: float = 2e-3
    lambda_T: float = 1e-4
    gate_cap_ns: float | None = 80.0


def cost_J(A, freq, settings, weights: CostWeights) -> Tuple[float, Dict[str, Any]]:
    """Placeholder cost function returning zeros.

    Args:
        A: Pulse amplitudes.
        freq: Frequency grid.
        settings: Control settings.
        weights: Weights for cost components.

    Returns:
        A tuple of (total_cost, parts_dict).
    """
    return 0.0, {}
