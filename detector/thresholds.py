"""Detection gate — decides whether a frame passes quality and signal thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GateDecision(str, Enum):
    PASS = "pass"
    REJECT_LOW_SIGNAL = "reject_low_signal"
    REJECT_HIGH_NOISE = "reject_high_noise"
    REJECT_OUT_OF_RANGE = "reject_out_of_range"


@dataclass(frozen=True)
class ThresholdConfig:
    """Minimum signal and quality requirements for a valid detection."""

    min_mean_intensity: float = 40.0
    max_mean_intensity: float = 220.0
    min_contrast_std: float = 12.0
    max_noise_estimate: float = 18.0
    min_hot_pixel_fraction: float = 0.002


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    mean_intensity: float
    contrast_std: float
    noise_estimate: float
    hot_pixel_fraction: float

    @property
    def passed(self) -> bool:
        return self.decision == GateDecision.PASS


class DetectionGate:
    """Evaluate preprocessed frames against device acceptance criteria."""

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()

    def evaluate(self, image: np.ndarray) -> GateResult:
        mean_intensity = float(np.mean(image))
        contrast_std = float(np.std(image))
        noise_estimate = self._estimate_noise(image)
        hot_pixel_fraction = self._hot_pixel_fraction(image)

        decision = self._decide(
            mean_intensity=mean_intensity,
            contrast_std=contrast_std,
            noise_estimate=noise_estimate,
            hot_pixel_fraction=hot_pixel_fraction,
        )

        return GateResult(
            decision=decision,
            mean_intensity=mean_intensity,
            contrast_std=contrast_std,
            noise_estimate=noise_estimate,
            hot_pixel_fraction=hot_pixel_fraction,
        )

    def _decide(
        self,
        *,
        mean_intensity: float,
        contrast_std: float,
        noise_estimate: float,
        hot_pixel_fraction: float,
    ) -> GateDecision:
        cfg = self.config

        if not (cfg.min_mean_intensity <= mean_intensity <= cfg.max_mean_intensity):
            return GateDecision.REJECT_OUT_OF_RANGE
        if contrast_std < cfg.min_contrast_std:
            return GateDecision.REJECT_LOW_SIGNAL
        if noise_estimate > cfg.max_noise_estimate:
            return GateDecision.REJECT_HIGH_NOISE
        if hot_pixel_fraction < cfg.min_hot_pixel_fraction:
            return GateDecision.REJECT_LOW_SIGNAL
        return GateDecision.PASS

    @staticmethod
    def _estimate_noise(image: np.ndarray) -> float:
        """Laplacian-based noise estimate (common in imaging QC)."""
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        from numpy.lib.stride_tricks import sliding_window_view

        if image.shape[0] < 3 or image.shape[1] < 3:
            return float(np.std(image))

        patches = sliding_window_view(image, (3, 3))
        responses = np.tensordot(patches, laplacian, axes=([2, 3], [0, 1]))
        return float(np.median(np.abs(responses)) / 0.6745)

    @staticmethod
    def _hot_pixel_fraction(image: np.ndarray, sigma: float = 3.0) -> float:
        median = float(np.median(image))
        mad = float(np.median(np.abs(image - median)))
        if mad == 0:
            return 0.0
        threshold = median + sigma * mad * 1.4826
        return float(np.mean(image > threshold))
