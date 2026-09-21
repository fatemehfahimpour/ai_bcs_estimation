"""Two-Stage Inference Engine for Cow Body Condition Score (BCS) Estimation.

Stage 1: Detects the anatomical region of the cow using YOLO11m.
Stage 2: Crops the region and predicts the BCS score using OrdinalResNet18.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from PIL import Image

# Preprocessing utilities
from data.preprocess import (
    INDEX_TO_BCS,
    PROJECT_ROOT,
    TARGET_SIZE,
    get_val_transform,
)

# Ordinal ResNet Architecture
from ReNet18_ordinal_classification_model.ordinal_model import OrdinalResNet18

# YOLO Detector
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class BCSInferenceEngine:
    """End-to-end Two-Stage BCS Inference Engine (Ordinal Version)."""

    def __init__(
        self,
        classifier_weights: Union[str, Path] = PROJECT_ROOT / "ReNet18_ordinal_classification_model" / "model_results_round2" / "best_ordinal_resnet18.pth",
        detector_weights: Optional[Union[str, Path]] = PROJECT_ROOT / "anatomical_region_ditector" / "runs" / "detect" / "anatomical_region_yolo11m" / "weights" / "best.pt",
        device: Optional[str] = None,
    ) -> None:
        """Initialize models, devices, and preprocessing transforms."""

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Setup Classifier (OrdinalResNet18)
        raw_cls = Path(classifier_weights)
        self.classifier_path = raw_cls if raw_cls.is_absolute() else PROJECT_ROOT / raw_cls
        self.transform = get_val_transform(target_size=TARGET_SIZE)
        self.classifier = self._load_classifier()

        # 2. Setup Detector (YOLO11m)
        self.detector = None

        if detector_weights is not None:
            raw_det = Path(detector_weights)
            self.detector_path = raw_det if raw_det.is_absolute() else PROJECT_ROOT / raw_det

            if self.detector_path.exists() and YOLO is not None:
                self.detector = YOLO(str(self.detector_path))

    def _load_classifier(self) -> OrdinalResNet18:
        """Load trained OrdinalResNet18 classifier state dict."""

        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"[!] Ordinal Classifier checkpoint not found at: {self.classifier_path.resolve()}"
            )

        checkpoint = torch.load(self.classifier_path, map_location=self.device)

        # Extract trainable_layers configuration if available
        trainable_layers = "fc"

        if isinstance(checkpoint, dict) and "trainable_layers" in checkpoint:
            trainable_layers = checkpoint["trainable_layers"]

        model = OrdinalResNet18(
            trainable_layers=trainable_layers,
            num_thresholds=4,
        )

        # Extract model state dictionary
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "best_model_state" in checkpoint:
                state_dict = checkpoint["best_model_state"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    @torch.inference_mode()
    def predict_crop(
        self,
        crop_image: Union[str, Path, Image.Image],
    ) -> Tuple[float, float, Dict[str, float]]:
        """Run Ordinal BCS classification on an already cropped anatomical image."""

        # Load image
        if isinstance(crop_image, (str, Path)):
            p = Path(crop_image)
            p = p if p.is_absolute() else PROJECT_ROOT / p

            if not p.exists():
                raise FileNotFoundError(f"Crop image not found: {p}")

            with Image.open(p) as img:
                pil_image = img.convert("RGB")

        elif isinstance(crop_image, Image.Image):
            pil_image = crop_image.convert("RGB")

        else:
            raise TypeError(f"Unsupported input type: {type(crop_image)}")

        # Transform image
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        # Ordinal outputs logits for 4 binary thresholds
        logits = self.classifier(tensor)

        probs = torch.sigmoid(logits).squeeze(0)

        # Standard ordinal prediction
        # Number of thresholds passed -> class index 0 to 4
        predicted_idx = int((probs > 0.5).sum().item())

        predicted_bcs = INDEX_TO_BCS.get(
            predicted_idx,
            float(predicted_idx),
        )

        # Expected continuous score
        expected_score = float(probs.sum().item())

        # Threshold probabilities
        threshold_details = {
            f"Threshold > {INDEX_TO_BCS.get(i, i)}": round(float(p.item()) * 100, 2)
            for i, p in enumerate(probs)
        }

        # Average certainty across thresholds
        certainty = float(
            torch.abs(probs - 0.5).mean().item() * 2 * 100
        )

        return (
            predicted_bcs,
            round(certainty, 2),
            threshold_details,
        )

    def predict_full_pipeline(
        self,
        full_image: Union[str, Path, Image.Image],
        conf_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Full Two-Stage Pipeline: Detect anatomical bbox -> Crop -> Predict BCS."""

        if self.detector is None:
            raise RuntimeError(
                "YOLO detector is not initialized. "
                "Provide valid detector_weights or use predict_crop()."
            )

        # Load full image
        if isinstance(full_image, (str, Path)):
            p = Path(full_image)
            p = p if p.is_absolute() else PROJECT_ROOT / p

            if not p.exists():
                raise FileNotFoundError(f"Full image not found: {p}")

            with Image.open(p) as img:
                pil_full = img.convert("RGB")

        elif isinstance(full_image, Image.Image):
            pil_full = full_image.convert("RGB")

        else:
            raise TypeError(f"Unsupported input type: {type(full_image)}")

        # Stage 1: Detect Anatomical Region
        det_results = self.detector.predict(
            source=pil_full,
            conf=conf_threshold,
            verbose=False,
        )

        boxes = det_results[0].boxes

        # No detection -> use full image
        if len(boxes) == 0:
            predicted_bcs, conf, probs = self.predict_crop(pil_full)

            return {
                "detected": False,
                "bbox": None,
                "crop_image": pil_full,
                "predicted_bcs": predicted_bcs,
                "confidence": conf,
                "threshold_probabilities": probs,
            }

        # Select highest confidence detection
        best_box = boxes[0]

        x1, y1, x2, y2 = map(
            int,
            best_box.xyxy[0].tolist(),
        )

        crop = pil_full.crop((x1, y1, x2, y2))

        # Stage 2: Classify BCS
        predicted_bcs, conf, probs = self.predict_crop(crop)

        return {
            "detected": True,
            "bbox": [x1, y1, x2, y2],
            "detector_conf": float(best_box.conf[0].item()),
            "crop_image": crop,
            "predicted_bcs": predicted_bcs,
            "confidence": conf,
            "threshold_probabilities": probs,
        }


if __name__ == "__main__":
    print("[*] Ordinal BCS Two-Stage Inference Engine loaded successfully.")