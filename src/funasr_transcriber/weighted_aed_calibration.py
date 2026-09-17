"""Device-specific weights for deterministic weighted AED scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class WeightedAedWeights:
    longest_weight: float
    padded_weight: float

    def __post_init__(self) -> None:
        values = (float(self.longest_weight), float(self.padded_weight))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "weighted AED coefficients must be finite and positive"
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "longest_weight": float(self.longest_weight),
            "padded_weight": float(self.padded_weight),
        }


DEFAULT_WEIGHTED_AED_WEIGHTS = WeightedAedWeights(
    longest_weight=0.12854663332839417,
    padded_weight=0.008409640427755745,
)
