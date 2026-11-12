"""Load the packaged RCS detector and score an image + question.

Artifact layout (data/outputs/rcs_artifact_v2):
  projection_layer9.pt   LearnedProjection weights
  kcd_state.npz          benign/malicious normalised vectors, k, threshold, layer
  meta.json              layer, feature config, metrics, model path
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rcs_features import QwenHiddenStateExtractor
from rcs_projection import load_projection, project_vectors

DEFAULT_ARTIFACT = Path(r"D:\CodexProjects\PrivaSee\data\outputs\rcs_artifact_v2")


class RCSDetector:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT):
        self.artifact_dir = Path(artifact_dir)
        self.meta = json.loads((self.artifact_dir / "meta.json").read_text(encoding="utf-8"))
        self.model, self.projection_meta = load_projection(self.artifact_dir / self.meta["projection"])
        state = np.load(self.artifact_dir / "kcd_state.npz")
        with state:
            self.benign = state["benign"].copy()
            self.malicious = state["malicious"].copy()
            self.k = int(state["k"])
            self.threshold = float(state["threshold"])
            self.layer = int(state["layer"])
        self.threshold_safe = float(self.meta.get("threshold_safe", self.threshold))
        self._validate()
        self._extractor = None

    def _validate(self) -> None:
        projection_layer = int(self.projection_meta.get("layer", -1))
        if projection_layer != self.layer:
            raise ValueError(f"投影层号 {projection_layer} 与参考库层号 {self.layer} 不一致")
        meta_layer = self.meta.get("layer")
        if meta_layer is not None and int(meta_layer) != self.layer:
            raise ValueError(f"meta 层号 {meta_layer} 与参考库层号 {self.layer} 不一致")
        architecture = self.projection_meta.get("architecture") or {}
        if int(architecture.get("input_dim", -1)) != 2048:
            raise ValueError(f"投影输入维度应为 2048，实际 {architecture.get('input_dim')}")
        if (self.benign.ndim != 2 or self.malicious.ndim != 2
                or self.benign.shape[1] != self.malicious.shape[1]):
            raise ValueError(f"参考库形状不匹配: benign={self.benign.shape}, malicious={self.malicious.shape}")
        if int(architecture.get("output_dim", -1)) != self.benign.shape[1]:
            raise ValueError(f"投影输出维度 {architecture.get('output_dim')} 与参考库维度 {self.benign.shape[1]} 不一致")
        if not (np.isfinite(self.benign).all() and np.isfinite(self.malicious).all()):
            raise ValueError("参考库包含 NaN/Inf")
        if not np.isfinite(self.threshold):
            raise ValueError("阈值不是有限值")
        if not (1 <= self.k <= min(len(self.benign), len(self.malicious))):
            raise ValueError(f"k={self.k} 超出参考库大小")
        meta_fingerprint = self.meta.get("feature_config_fingerprint")
        projection_fingerprint = self.projection_meta.get("feature_config_fingerprint")
        if meta_fingerprint and projection_fingerprint and meta_fingerprint != projection_fingerprint:
            raise ValueError("meta 与投影的特征配置指纹不一致")

    def _normalise(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def score_projected(self, projected: np.ndarray) -> float:
        z = self._normalise(projected)
        d_benign = np.sort(np.linalg.norm(self.benign - z, axis=1))[min(self.k, len(self.benign)) - 1]
        d_malicious = np.sort(np.linalg.norm(self.malicious - z, axis=1))[min(self.k, len(self.malicious)) - 1]
        return float(d_benign - d_malicious)

    def score_vector(self, hidden: np.ndarray) -> float:
        if np.asarray(hidden).reshape(-1).shape[0] != 2048:
            raise ValueError("hidden 必须是 2048 维")
        return self.score_projected(project_vectors(self.model, np.asarray(hidden).reshape(1, -1))[0])

    def extractor(self) -> QwenHiddenStateExtractor:
        if self._extractor is None:
            self._extractor = QwenHiddenStateExtractor(
                min_visual_tokens=self.meta.get("min_visual_tokens", 256),
                max_visual_tokens=self.meta.get("max_visual_tokens", 768),
            )
            self._extractor.load()
        return self._extractor

    def score_image(self, image: Path, question: str) -> dict:
        vectors, details = self.extractor().extract({"image": str(Path(image)), "question": question}, [self.layer])
        score = self.score_vector(vectors[0])
        return {
            "image": str(image),
            "question": question,
            "layer": self.layer,
            "k": self.k,
            "score": score,
            "threshold_safe": self.threshold_safe,
            "threshold_calibrated": self.threshold,
            "decision": "safe" if score <= self.threshold_safe else "suspicious",
            "token_index": details.get("token_index"),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--threshold-safe", type=float, default=None,
                        help="覆盖 artifact 里的安全阈值（默认 -0.05）")
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    detector = RCSDetector(args.artifact)
    if args.threshold_safe is not None:
        detector.threshold_safe = args.threshold_safe
    result = detector.score_image(args.image, args.question)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
