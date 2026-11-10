"""
YOLO object detection module for warehouse perception.
"""

from pathlib import Path

import torch
from ultralytics import YOLO


class WarehouseDetector:
    """Load a YOLO model and perform object detection."""

    def __init__(
        self,
        weights_path,
        confidence=0.25,
        device=None,
    ):
        self.weights_path = Path(weights_path)

        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)

        self.confidence = confidence
        self.device = device or (
            "0" if torch.cuda.is_available() else "cpu"
        )

        self.model = YOLO(str(self.weights_path))

    def detect(self, image, confidence=None):
        """
        Detect objects in an image.

        Parameters
        ----------
        image:
            Image path, NumPy array, or supported PIL image.

        confidence:
            Optional confidence threshold.

        Returns
        -------
        list[dict]:
            Structured detections with class, confidence and bbox.
        """

        threshold = (
            self.confidence if confidence is None else confidence
        )

        results = self.model.predict(
            source=image,
            conf=threshold,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": tuple(
                        float(value)
                        for value in box.xyxy[0].tolist()
                    ),
                })

        return detections

    def predict(self, image, confidence=None):
        """Return the original Ultralytics prediction results."""
        threshold = (
            self.confidence if confidence is None else confidence
        )

        return self.model.predict(
            source=image,
            conf=threshold,
            device=self.device,
            verbose=False,
        )
