"""
Person detection using YOLO.
"""

from typing import Any, Optional, List
import numpy as np

from src.models.base import BaseModel
from src.config.settings import ModelSettings


class PersonDetector(BaseModel):
    """
    YOLO-based person detection model wrapper.

    Uses Ultralytics YOLO for detecting persons in images.
    """

    def __init__(self, config: Optional[ModelSettings] = None):
        """
        Initialize the person detector.

        Args:
            config: Model configuration settings
        """
        super().__init__()
        self.config = config or ModelSettings()

    def load(self) -> None:
        """Load the YOLO detection model."""
        from ultralytics import YOLO

        self._model = YOLO(self.config.yolo_detect_model)
        self._is_loaded = True

    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Detect persons in an image.

        Args:
            image: Input image (BGR format)
            conf: Confidence threshold (uses config default if None)
            iou: IoU threshold for NMS (uses config default if None)
            **kwargs: Additional YOLO parameters

        Returns:
            YOLO detection results object
        """
        self.ensure_loaded()

        conf = conf if conf is not None else self.config.detection_confidence
        iou = iou if iou is not None else self.config.iou_threshold

        results = self._model(
            image,
            conf=conf,
            iou=iou,
            classes=self.config.person_class_id,
            verbose=False,
            **kwargs
        )

        return results[0]

    def get_boxes(self, image: np.ndarray, conf: Optional[float] = None) -> List[np.ndarray]:
        """
        Get bounding boxes for detected persons.

        Args:
            image: Input image
            conf: Confidence threshold

        Returns:
            List of bounding boxes in [x1, y1, x2, y2] format
        """
        results = self.predict(image, conf=conf)

        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                boxes.append(bbox)

        return boxes

    def get_detections_with_confidence(
        self,
        image: np.ndarray,
        conf: Optional[float] = None
    ) -> List[dict]:
        """
        Get detections with confidence scores.

        Args:
            image: Input image
            conf: Confidence threshold

        Returns:
            List of dictionaries with 'bbox' and 'confidence' keys
        """
        results = self.predict(image, conf=conf)

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence
                })

        return detections

    def count_persons(self, image: np.ndarray, conf: Optional[float] = None) -> int:
        """
        Count the number of persons in an image.

        Args:
            image: Input image
            conf: Confidence threshold

        Returns:
            Number of detected persons
        """
        results = self.predict(image, conf=conf)

        if results.boxes is not None:
            return len(results.boxes)
        return 0
