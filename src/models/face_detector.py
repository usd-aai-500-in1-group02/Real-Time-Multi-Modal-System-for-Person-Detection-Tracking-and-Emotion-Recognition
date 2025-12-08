"""
Face detection using MTCNN.
NEW: Dedicated face detector as per PDF requirements.
"""

from typing import Any, Optional, List, Tuple
import numpy as np

from src.models.base import BaseModel
from src.config.settings import ModelSettings


class FaceDetector(BaseModel):
    """
    MTCNN-based face detection model wrapper.

    Detects faces in images with landmarks for better emotion recognition.
    This is a dedicated face detector (as required by PDF) instead of
    using person bounding boxes directly.
    """

    def __init__(self, config: Optional[ModelSettings] = None):
        """
        Initialize the face detector.

        Args:
            config: Model configuration settings
        """
        super().__init__()
        self.config = config or ModelSettings()

    def load(self) -> None:
        """Load the MTCNN face detector."""
        from mtcnn import MTCNN

        self._model = MTCNN()
        self._is_loaded = True

    def predict(self, image: np.ndarray, **kwargs) -> List[dict]:
        """
        Detect faces in an image.

        Args:
            image: Input image (BGR format - will be converted to RGB)
            **kwargs: Additional parameters

        Returns:
            List of face detection dictionaries from MTCNN
        """
        self.ensure_loaded()

        # MTCNN expects RGB
        import cv2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self._model.detect_faces(image_rgb)

        return faces

    def get_face_boxes(
        self,
        image: np.ndarray,
        min_confidence: float = 0.9
    ) -> List[np.ndarray]:
        """
        Get bounding boxes for detected faces.

        Args:
            image: Input image
            min_confidence: Minimum confidence threshold

        Returns:
            List of bounding boxes in [x1, y1, x2, y2] format
        """
        faces = self.predict(image)

        boxes = []
        for face in faces:
            if face['confidence'] >= min_confidence:
                x, y, w, h = face['box']
                # Convert to [x1, y1, x2, y2]
                boxes.append(np.array([x, y, x + w, y + h]))

        return boxes

    def get_faces_with_landmarks(
        self,
        image: np.ndarray,
        min_confidence: float = 0.9
    ) -> List[dict]:
        """
        Get face detections with landmarks.

        Args:
            image: Input image
            min_confidence: Minimum confidence threshold

        Returns:
            List of dictionaries with bbox, confidence, and landmarks
        """
        faces = self.predict(image)

        results = []
        for face in faces:
            if face['confidence'] >= min_confidence:
                x, y, w, h = face['box']

                results.append({
                    'bbox': np.array([x, y, x + w, y + h]),
                    'bbox_xywh': face['box'],
                    'confidence': face['confidence'],
                    'landmarks': face.get('keypoints', {})
                })

        return results

    def crop_faces(
        self,
        image: np.ndarray,
        min_confidence: float = 0.9,
        padding: float = 0.1
    ) -> List[Tuple[np.ndarray, dict]]:
        """
        Crop face regions from image.

        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            padding: Padding ratio around face (0.1 = 10% on each side)

        Returns:
            List of tuples (cropped_face_image, face_info_dict)
        """
        faces = self.get_faces_with_landmarks(image, min_confidence)
        h, w = image.shape[:2]

        crops = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']

            # Add padding
            pad_w = (x2 - x1) * padding
            pad_h = (y2 - y1) * padding

            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(w, int(x2 + pad_w))
            y2 = min(h, int(y2 + pad_h))

            cropped = image[y1:y2, x1:x2].copy()

            if cropped.size > 0:
                crops.append((cropped, face))

        return crops

    def detect_faces_in_person_bbox(
        self,
        image: np.ndarray,
        person_bbox: np.ndarray,
        min_confidence: float = 0.8
    ) -> List[dict]:
        """
        Detect faces within a person bounding box region.

        Args:
            image: Full image
            person_bbox: Person bounding box [x1, y1, x2, y2]
            min_confidence: Minimum confidence threshold

        Returns:
            List of face detections with coordinates relative to full image
        """
        x1, y1, x2, y2 = map(int, person_bbox[:4])

        # Crop person region
        person_crop = image[y1:y2, x1:x2]

        if person_crop.size == 0:
            return []

        # Detect faces in crop
        faces = self.get_faces_with_landmarks(person_crop, min_confidence)

        # Adjust coordinates to full image
        for face in faces:
            face['bbox'] = face['bbox'] + np.array([x1, y1, x1, y1])
            face['person_bbox'] = person_bbox

        return faces
