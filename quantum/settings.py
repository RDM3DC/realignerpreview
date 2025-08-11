from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class ARPSettings:
    """Minimal settings for quantum control experiments."""
    T1: Sequence[float]
    Tphi: Sequence[float]
    Delta: Sequence[float]
    pairs: Iterable[tuple[int, int]] = ((0, 1), (2, 3))
    xt_bands: Iterable[tuple[float, float]] = ()
