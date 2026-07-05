"""End-to-end detection pipeline for the Galen demo device."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from detector.preprocess import PreprocessConfig, Preprocessor
from detector.thresholds import DetectionGate, GateResult, ThresholdConfig


@dataclass(frozen=True)
class PipelineResult:
    gate: GateResult
    processed_frame: np.ndarray
    metadata: dict[str, Any]


class DetectionPipeline:
    """Load config, preprocess frames, and run the detection gate."""

    def __init__(
        self,
        preprocess: Preprocessor | None = None,
        gate: DetectionGate | None = None,
    ) -> None:
        self.preprocessor = preprocess or Preprocessor()
        self.gate = gate or DetectionGate()

    @classmethod
    def from_config(cls, config_path: str | Path) -> DetectionPipeline:
        path = Path(config_path)
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        preprocess_cfg = PreprocessConfig(**raw.get("preprocess", {}))
        threshold_cfg = ThresholdConfig(**raw.get("thresholds", {}))
        return cls(
            preprocess=Preprocessor(preprocess_cfg),
            gate=DetectionGate(threshold_cfg),
        )

    def process(self, frame: np.ndarray, *, frame_id: str | None = None) -> PipelineResult:
        processed = self.preprocessor.run(frame)
        gate_result = self.gate.evaluate(processed)
        metadata = {
            "frame_id": frame_id,
            "passed": gate_result.passed,
            "decision": gate_result.decision.value,
        }
        return PipelineResult(
            gate=gate_result,
            processed_frame=processed,
            metadata=metadata,
        )
