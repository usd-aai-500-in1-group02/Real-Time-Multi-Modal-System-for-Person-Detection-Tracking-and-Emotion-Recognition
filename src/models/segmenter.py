"""
Person instance segmentation using YOLO-seg.
"""

from typing import Any, Optional, List, Tuple
import numpy as np
import cv2

from src.models.base import BaseModel
from src.config.settings import ModelSettings


class PersonSegmenter(BaseModel):
    """
    YOLO-based person instance segmentation model wrapper.

    Uses Ultralytics YOLO-seg for pixel-level person segmentation.
    """

    def __init__(self, config: Optional[ModelSettings] = None):
        """
        Initialize the person segmenter.

        Args:
            config: Model configuration settings
        """
        super().__init__()
        self.config = config or ModelSettings()

    def load(self) -> None:
        """Load the YOLO segmentation model."""
        from ultralytics import YOLO

        self._model = YOLO(self.config.yolo_segment_model)
        self._is_loaded = True

    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Segment persons in an image.

        Args:
            image: Input image (BGR format)
            conf: Confidence threshold (uses config default if None)
            iou: IoU threshold for NMS (uses config default if None)
            **kwargs: Additional YOLO parameters

        Returns:
            YOLO segmentation results object
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

    def get_masks(
        self,
        image: np.ndarray,
        conf: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Get segmentation masks for detected persons.

        Args:
            image: Input image
            conf: Confidence threshold

        Returns:
            List of binary masks (same size as input image)
        """
        results = self.predict(image, conf=conf)

        masks = []
        if hasattr(results, 'masks') and results.masks is not None:
            mask_data = results.masks.data.cpu().numpy()
            for mask in mask_data:
                # Resize mask to image size
                mask_resized = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                masks.append(mask_resized)

        return masks

    def get_masks_with_boxes(
        self,
        image: np.ndarray,
        conf: Optional[float] = None
    ) -> List[dict]:
        """
        Get segmentation masks with corresponding bounding boxes.

        Args:
            image: Input image
            conf: Confidence threshold

        Returns:
            List of dictionaries with 'mask', 'bbox', and 'confidence' keys
        """
        results = self.predict(image, conf=conf)

        segments = []
        if hasattr(results, 'masks') and results.masks is not None and results.boxes is not None:
            mask_data = results.masks.data.cpu().numpy()

            for i, (mask, box) in enumerate(zip(mask_data, results.boxes)):
                mask_resized = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                segments.append({
                    'mask': mask_resized,
                    'bbox': bbox,
                    'confidence': confidence
                })

        return segments

    def create_colored_overlay(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        color: Tuple[int, int, int] = (255, 0, 255),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Create a colored overlay from segmentation masks.

        Args:
            image: Original image
            masks: List of binary masks
            color: BGR color for overlay
            alpha: Transparency (0-1)

        Returns:
            Image with colored mask overlay
        """
        overlay = image.copy()

        for mask in masks:
            colored_mask = np.zeros_like(image)
            colored_mask[:, :] = color

            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            overlay = np.where(
                mask_3ch > 0.5,
                cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0),
                overlay
            )

        return overlay
