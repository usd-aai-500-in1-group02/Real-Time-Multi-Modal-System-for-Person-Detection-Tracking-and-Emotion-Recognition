"""
Multi-object tracking using DeepSort.
"""

from typing import Any, Optional, List
import numpy as np

from src.models.base import BaseModel
from src.config.settings import TrackerSettings


class PersonTracker(BaseModel):
    """
    DeepSort-based multi-object tracker wrapper.

    Maintains consistent person IDs across video frames.
    """

    def __init__(self, config: Optional[TrackerSettings] = None):
        """
        Initialize the person tracker.

        Args:
            config: Tracker configuration settings
        """
        super().__init__()
        self.config = config or TrackerSettings()

    def load(self) -> None:
        """Load the DeepSort tracker."""
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._model = DeepSort(
            max_age=self.config.max_age,
            n_init=self.config.n_init,
            nms_max_overlap=self.config.nms_max_overlap,
            max_cosine_distance=self.config.max_cosine_distance,
            nn_budget=self.config.nn_budget
        )
        self._is_loaded = True

    def predict(
        self,
        detections: Any,
        frame: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Any]:
        """
        Update tracker with new detections.

        Args:
            detections: YOLO detection results or list of detection dicts
            frame: Current video frame (required for Re-ID features)
            **kwargs: Additional tracker parameters

        Returns:
            List of confirmed tracks
        """
        self.ensure_loaded()

        detection_list = self._convert_detections(detections)
        tracks = self._model.update_tracks(detection_list, frame=frame)

        return tracks

    def _convert_detections(self, detections: Any) -> List:
        """
        Convert YOLO detections to DeepSort format.

        Args:
            detections: YOLO detection results or list of dicts

        Returns:
            List of detections in DeepSort format: ([x, y, w, h], conf, class)
        """
        detection_list = []

        # Handle YOLO detection results
        if hasattr(detections, 'boxes') and detections.boxes is not None:
            for box in detections.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                detection_list.append(([x1, y1, w, h], conf, 'person'))

        # Handle list of detection dicts
        elif isinstance(detections, list):
            for det in detections:
                if isinstance(det, dict):
                    bbox = det['bbox']
                    conf = det.get('confidence', 1.0)

                    x1, y1, x2, y2 = bbox[:4]
                    w, h = x2 - x1, y2 - y1

                    detection_list.append(([x1, y1, w, h], conf, 'person'))

        return detection_list

    def get_confirmed_tracks(self, tracks: List[Any]) -> List[dict]:
        """
        Get only confirmed tracks with their information.

        Args:
            tracks: List of tracks from predict()

        Returns:
            List of dictionaries with track information
        """
        confirmed = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()  # [x1, y1, x2, y2]

            confirmed.append({
                'track_id': track.track_id,
                'bbox': bbox,
                'center': (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ),
                'is_confirmed': True
            })

        return confirmed

    def reset(self) -> None:
        """Reset the tracker state."""
        if self._is_loaded:
            self.load()  # Reinitialize to clear all tracks
