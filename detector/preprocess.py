"""Image preprocessing and filtering for the Galen demo device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


FilterMode = Literal["gaussian", "median", "bilateral"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Tunable preprocessing parameters."""

    blur_kernel: int = 5
    blur_sigma: float = 1.2
    filter_mode: FilterMode = "gaussian"
    normalize: bool = True
    clip_percentile: tuple[float, float] = (1.0, 99.0)
    smoothing_enabled: bool = False
    smoothing_kernel: int = 3
    smoothing_sigma: float = 0.8


class Preprocessor:
    """Apply denoising, contrast normalization, and edge-preserving filters."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Return a filtered grayscale image ready for thresholding."""
        gray = self._to_grayscale(frame)
        if self.config.smoothing_enabled:
            gray = self._apply_smoothing(gray)
        filtered = self._apply_filter(gray)
        if self.config.normalize:
            filtered = self._normalize_contrast(filtered)
        return filtered

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float32)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _apply_smoothing(self, gray: np.ndarray) -> np.ndarray:
        """Optional pre-detection smoothing pass to suppress high-frequency noise."""
        k = self._odd_kernel(self.config.smoothing_kernel)
        return cv2.GaussianBlur(gray, (k, k), self.config.smoothing_sigma)

    def _apply_filter(self, gray: np.ndarray) -> np.ndarray:
        k = self._odd_kernel(self.config.blur_kernel)
        mode = self.config.filter_mode

        if mode == "gaussian":
            return cv2.GaussianBlur(gray, (k, k), self.config.blur_sigma)
        if mode == "median":
            return cv2.medianBlur(gray.astype(np.uint8), k).astype(np.float32)
        if mode == "bilateral":
            return cv2.bilateralFilter(
                gray.astype(np.uint8), d=k, sigmaColor=50, sigmaSpace=50
            ).astype(np.float32)
        raise ValueError(f"Unknown filter mode: {mode!r}")

    def _normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        lo, hi = self.config.clip_percentile
        lower = np.percentile(image, lo)
        upper = np.percentile(image, hi)
        if upper <= lower:
            return image
        clipped = np.clip(image, lower, upper)
        return ((clipped - lower) / (upper - lower) * 255.0).astype(np.float32)

    @staticmethod
    def _odd_kernel(size: int) -> int:
        size = max(3, int(size))
        return size if size % 2 == 1 else size + 1
