"""Two-Stage Inference Engine for Cow Body Condition Score (BCS) Estimation.

Stage 1: Detects the anatomical region of the cow using YOLO11m.
Stage 2: Crops the region and predicts the BCS score using BCSResNet18.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

# Import preprocessing specs
from data.preprocess import (
    INDEX_TO_BCS,
    PROJECT_ROOT,
    TARGET_SIZE,
    get_val_transform,
)

# Import ResNet architecture
from ResNet18_classification_model.ResNet_model import BCSResNet18

# Import YOLO for Stage 1 detection
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class BCSInferenceEngine:
    """End-to-end Two-Stage BCS Inference Engine."""

    def __init__(
        self,
        classifier_weights: Union[
            str, Path
        ] = "model_results/best_resnet18.pth",
        detector_weights: Optional[Union[str, Path]] = (
            "runs/detect/anatomical_region_yolo11m/weights/best.pt"
        ),
        device: Optional[str] = None,
    ) -> None:
        """Initialize models, devices, and preprocessing transforms."""
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # 1. Classifier Setup (ResNet18)
        raw_cls = Path(classifier_weights)
        self.classifier_path = (
            raw_cls if raw_cls.is_absolute() else PROJECT_ROOT / raw_cls
        )
        self.transform = get_val_transform(target_size=TARGET_SIZE)
        self.classifier = self._load_classifier()

        # 2. Detector Setup (YOLO11m) - Optional / Lazy-loaded
        self.detector = None
        if detector_weights is not None:
            raw_det = Path(detector_weights)
            self.detector_path = (
                raw_det if raw_det.is_absolute() else PROJECT_ROOT / raw_det
            )
            if self.detector_path.exists() and YOLO is not None:
                self.detector = YOLO(str(self.detector_path))

    def _load_classifier(self) -> BCSResNet18:
        """Load trained ResNet18 classifier state dict."""
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"[!] Classifier checkpoint not found at: {self.classifier_path.resolve()}"
            )

        checkpoint = torch.load(self.classifier_path, map_location=self.device)

        # Extract trainable_layers configuration if available
        trainable_layers = "fc"
        if isinstance(checkpoint, dict) and "trainable_layers" in checkpoint:
            trainable_layers = checkpoint["trainable_layers"]

        model = BCSResNet18(trainable_layers=trainable_layers)

        # Extract state dict from ResNet_hyperparameter_search.py format
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "best_model_state" in checkpoint:
                state_dict = checkpoint["best_model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Ensure correct prefix matching
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith("model."):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    @torch.inference_mode()
    def predict_crop(
        self, crop_image: Union[str, Path, Image.Image]
    ) -> Tuple[float, float, Dict[float, float]]:
        """Run BCS classification on an already cropped anatomical image."""
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

        # Gray Padding (MakeSquareWithGrayPadding) -> Resize 224x224 -> ImageNet Norm
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        logits = self.classifier(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        best_idx = int(torch.argmax(probs).item())
        confidence = float(probs[best_idx].item())
        predicted_bcs = INDEX_TO_BCS[best_idx]

        class_probabilities = {
            INDEX_TO_BCS[i]: round(float(p.item()) * 100, 2)
            for i, p in enumerate(probs)
        }

        return predicted_bcs, confidence, class_probabilities

    def predict_full_pipeline(
        self,
        full_image: Union[str, Path, Image.Image],
        conf_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Full Two-Stage Pipeline: Detect anatomical bbox -> Crop -> Predict BCS."""
        if self.detector is None:
            raise RuntimeError(
                "YOLO detector is not initialized. Provide valid detector_weights or use predict_crop()."
            )

        if isinstance(full_image, (str, Path)):
            p = Path(full_image)
            p = p if p.is_absolute() else PROJECT_ROOT / p
            with Image.open(p) as img:
                pil_full = img.convert("RGB")
        else:
            pil_full = full_image.convert("RGB")

        # Stage 1: Detect Anatomical Region
        det_results = self.detector.predict(
            source=pil_full, conf=conf_threshold, verbose=False
        )
        boxes = det_results[0].boxes

        if len(boxes) == 0:
            # Fallback: if no box is detected, classify whole image
            predicted_bcs, conf, probs = self.predict_crop(pil_full)
            return {
                "detected": False,
                "bbox": None,
                "crop_image": pil_full,
                "predicted_bcs": predicted_bcs,
                "confidence": conf,
                "class_probabilities": probs,
            }

        # Select highest confidence box
        best_box = boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
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
            "class_probabilities": probs,
        }


if __name__ == "__main__":
    print("[*] BCS Two-Stage Inference Engine loaded successfully.")
