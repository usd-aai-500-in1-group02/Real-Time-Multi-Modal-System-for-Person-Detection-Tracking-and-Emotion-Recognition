"""
Emotion recognition using DeepFace.
"""

from typing import Any, Optional, Dict, List
import numpy as np

from src.models.base import BaseModel
from src.config.settings import ModelSettings


class EmotionRecognizer(BaseModel):
    """
    DeepFace-based emotion recognition model wrapper.

    Classifies facial expressions into 7 emotion categories:
    happy, sad, angry, neutral, surprise, fear, disgust
    """

    EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']

    def __init__(self, config: Optional[ModelSettings] = None):
        """
        Initialize the emotion recognizer.

        Args:
            config: Model configuration settings
        """
        super().__init__()
        self.config = config or ModelSettings()

    def load(self) -> None:
        """
        Initialize DeepFace.

        DeepFace loads models on first use, so we just mark as loaded.
        """
        # DeepFace is imported on demand
        self._model = True  # Placeholder
        self._is_loaded = True

    def predict(
        self,
        image: np.ndarray,
        face_bbox: Optional[np.ndarray] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Recognize emotion from a face image or face region.

        Args:
            image: Input image (BGR format) - full image or cropped face
            face_bbox: Optional face bounding box [x1, y1, x2, y2]
            **kwargs: Additional parameters

        Returns:
            Dictionary with emotion results or None if detection fails
        """
        self.ensure_loaded()

        try:
            from deepface import DeepFace

            # If bbox provided, crop to face region
            if face_bbox is not None:
                x1, y1, x2, y2 = map(int, face_bbox[:4])
                face_img = image[y1:y2, x1:x2]
                if face_img.size == 0:
                    return None
            else:
                face_img = image

            # Analyze emotion
            analysis = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.config.emotion_detector_backend,
                silent=True
            )

            # Handle list response
            if isinstance(analysis, list):
                analysis = analysis[0]

            dominant_emotion = analysis['dominant_emotion']
            emotion_scores = analysis['emotion']

            return {
                'emotion': dominant_emotion,
                'confidence': emotion_scores[dominant_emotion] / 100.0,
                'all_emotions': {k: v / 100.0 for k, v in emotion_scores.items()},
                'face_bbox': face_bbox
            }

        except Exception as e:
            # Return None if emotion detection fails
            return None

    def analyze_multiple_faces(
        self,
        image: np.ndarray,
        face_bboxes: List[np.ndarray]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Analyze emotions for multiple faces.

        Args:
            image: Full image
            face_bboxes: List of face bounding boxes

        Returns:
            List of emotion results (None for failed detections)
        """
        results = []
        for bbox in face_bboxes:
            result = self.predict(image, face_bbox=bbox)
            results.append(result)
        return results

    def get_dominant_emotion(
        self,
        image: np.ndarray,
        face_bbox: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Get only the dominant emotion label.

        Args:
            image: Input image
            face_bbox: Optional face bounding box

        Returns:
            Emotion label string or None
        """
        result = self.predict(image, face_bbox)
        if result:
            return result['emotion']
        return None

    def get_emotion_distribution(
        self,
        image: np.ndarray,
        face_bbox: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get full emotion probability distribution.

        Args:
            image: Input image
            face_bbox: Optional face bounding box

        Returns:
            Dictionary mapping emotions to confidence scores
        """
        result = self.predict(image, face_bbox)
        if result:
            return result['all_emotions']
        return None
