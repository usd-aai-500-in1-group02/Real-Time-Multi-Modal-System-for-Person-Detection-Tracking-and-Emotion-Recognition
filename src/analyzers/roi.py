"""
Region of Interest (ROI) management.
Extracted from app4.py lines 530-573.
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import numpy as np
import cv2

from src.analyzers.base import BaseAnalyzer


class ROIManager(BaseAnalyzer):
    """
    Manage Regions of Interest for zone-based analysis.

    Allows defining polygon zones and counting/tracking
    persons within those zones.
    """

    def __init__(self):
        """Initialize the ROI manager."""
        super().__init__()
        self.rois: List[Dict[str, Any]] = []
        self.roi_stats: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {'detections': 0, 'entries': 0, 'current_count': 0}
        )
        self._previous_in_roi: Dict[int, set] = defaultdict(set)
        self._is_initialized = True

    def add_roi(
        self,
        name: str,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        active: bool = True
    ) -> int:
        """
        Add a new ROI.

        Args:
            name: Display name for the ROI
            points: List of polygon vertices as (x, y) tuples
            color: BGR color for visualization
            active: Whether ROI is active

        Returns:
            Index of the new ROI
        """
        roi = {
            'name': name,
            'points': np.array(points, dtype=np.int32),
            'color': color,
            'active': active
        }
        self.rois.append(roi)
        return len(self.rois) - 1

    def remove_roi(self, roi_idx: int) -> bool:
        """
        Remove an ROI by index.

        Args:
            roi_idx: Index of ROI to remove

        Returns:
            True if removed successfully
        """
        if 0 <= roi_idx < len(self.rois):
            self.rois.pop(roi_idx)
            return True
        return False

    def is_point_in_roi(
        self,
        point: Tuple[float, float],
        roi_idx: int
    ) -> bool:
        """
        Check if a point is inside an ROI.

        Args:
            point: Point as (x, y)
            roi_idx: Index of ROI

        Returns:
            True if point is inside ROI
        """
        if roi_idx >= len(self.rois):
            return False

        roi = self.rois[roi_idx]
        if not roi['active']:
            return False

        result = cv2.pointPolygonTest(roi['points'], tuple(map(int, point)), False)
        return result >= 0

    def update(self, detections: Any, roi_idx: Optional[int] = None) -> Dict[int, int]:
        """
        Update ROI counts from detections.

        Args:
            detections: YOLO detection results
            roi_idx: Specific ROI index or None for all

        Returns:
            Dictionary mapping ROI index to count
        """
        counts = {}
        roi_indices = [roi_idx] if roi_idx is not None else range(len(self.rois))

        for idx in roi_indices:
            count = self.count_in_roi(detections, idx)
            counts[idx] = count

        return counts

    def count_in_roi(self, detections: Any, roi_idx: int) -> int:
        """
        Count detections inside an ROI.

        Args:
            detections: YOLO detection results
            roi_idx: Index of ROI

        Returns:
            Number of detections in ROI
        """
        if detections is None or not hasattr(detections, 'boxes') or detections.boxes is None:
            self.roi_stats[roi_idx]['current_count'] = 0
            return 0

        count = 0
        for box in detections.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if self.is_point_in_roi(center, roi_idx):
                count += 1
                self.roi_stats[roi_idx]['detections'] += 1

        self.roi_stats[roi_idx]['current_count'] = count
        return count

    def update_from_tracks(self, tracks: List[Any]) -> Dict[int, List[int]]:
        """
        Update ROI statistics from tracks (for entry counting).

        Args:
            tracks: List of track objects

        Returns:
            Dictionary mapping ROI index to list of track IDs inside
        """
        roi_tracks = defaultdict(list)

        for roi_idx in range(len(self.rois)):
            current_in_roi = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                if self.is_point_in_roi(center, roi_idx):
                    current_in_roi.add(track_id)
                    roi_tracks[roi_idx].append(track_id)

                    # Check for new entry
                    if track_id not in self._previous_in_roi[roi_idx]:
                        self.roi_stats[roi_idx]['entries'] += 1

            self._previous_in_roi[roi_idx] = current_in_roi
            self.roi_stats[roi_idx]['current_count'] = len(current_in_roi)

        return dict(roi_tracks)

    def draw(self, image: np.ndarray) -> np.ndarray:
        """
        Draw ROI boundaries and labels on image.

        Args:
            image: Input image

        Returns:
            Image with ROI overlays
        """
        img_draw = image.copy()

        for idx, roi in enumerate(self.rois):
            if not roi['active']:
                continue

            # Draw polygon
            cv2.polylines(img_draw, [roi['points']], True, roi['color'], 3)

            # Draw name and count
            center = np.mean(roi['points'], axis=0).astype(int)
            count = self.roi_stats[idx]['current_count']

            cv2.putText(
                img_draw,
                f"{roi['name']}: {count}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                roi['color'],
                2
            )

        return img_draw

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ROI statistics.

        Returns:
            Dictionary with statistics per ROI
        """
        stats = {}
        for idx, roi in enumerate(self.rois):
            stats[roi['name']] = {
                'index': idx,
                'active': roi['active'],
                **self.roi_stats[idx]
            }
        return stats

    def get_roi_info(self, roi_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific ROI.

        Args:
            roi_idx: Index of ROI

        Returns:
            ROI information dictionary or None
        """
        if 0 <= roi_idx < len(self.rois):
            roi = self.rois[roi_idx]
            return {
                'name': roi['name'],
                'points': roi['points'].tolist(),
                'color': roi['color'],
                'active': roi['active'],
                **self.roi_stats[roi_idx]
            }
        return None

    def set_roi_active(self, roi_idx: int, active: bool) -> None:
        """Set whether an ROI is active."""
        if 0 <= roi_idx < len(self.rois):
            self.rois[roi_idx]['active'] = active

    def reset(self) -> None:
        """Reset all ROI statistics."""
        self.roi_stats.clear()
        self._previous_in_roi.clear()
