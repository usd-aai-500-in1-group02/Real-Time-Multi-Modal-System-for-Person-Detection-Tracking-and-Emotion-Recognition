"""
Social distancing monitoring.
Extracted from app4.py lines 33-111.
"""

from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2
from scipy.spatial import distance as dist

from src.analyzers.base import BaseAnalyzer
from src.utils.geometry import calculate_bottom_centroid


class SocialDistancingMonitor(BaseAnalyzer):
    """
    Monitor social distancing violations in real-time.

    Calculates pairwise distances between detected persons
    and identifies violations when distance is below threshold.
    """

    def __init__(self, min_distance: int = 150, image_height: int = 720):
        """
        Initialize the social distancing monitor.

        Args:
            min_distance: Minimum safe distance in pixels
            image_height: Reference image height for scaling
        """
        super().__init__()
        self.min_distance = min_distance
        self.image_height = image_height
        self.violations: List[Dict[str, Any]] = []
        self.violation_history: Dict[str, int] = defaultdict(int)
        self._total_checks = 0
        self._is_initialized = True

    def update(self, detections: Any) -> List[Dict[str, Any]]:
        """
        Calculate distances and detect violations.

        Args:
            detections: YOLO detection results

        Returns:
            List of violation dictionaries
        """
        self.violations = self.calculate_distances(detections)
        return self.violations

    def calculate_distances(self, detections: Any) -> List[Dict[str, Any]]:
        """
        Calculate pairwise distances between all detected persons.

        Args:
            detections: YOLO detection results

        Returns:
            List of violation dictionaries with pair info
        """
        if detections is None or not hasattr(detections, 'boxes') or detections.boxes is None:
            return []

        if len(detections.boxes) < 2:
            return []

        self._total_checks += 1

        # Calculate bottom centroids (feet position for ground-plane distance)
        centroids = []
        for box in detections.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            centroid = calculate_bottom_centroid(bbox)
            centroids.append(centroid)

        centroids = np.array(centroids)
        violation_pairs = []

        # Check all pairs
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = dist.euclidean(centroids[i], centroids[j])

                if distance < self.min_distance:
                    pair_key = f"{min(i,j)}-{max(i,j)}"
                    self.violation_history[pair_key] += 1

                    violation_pairs.append({
                        'pair': (i, j),
                        'distance': distance,
                        'centroid1': tuple(centroids[i]),
                        'centroid2': tuple(centroids[j]),
                        'severity': self._calculate_severity(distance)
                    })

        self.violations = violation_pairs
        return violation_pairs

    def _calculate_severity(self, distance: float) -> str:
        """Calculate violation severity based on distance."""
        ratio = distance / self.min_distance
        if ratio < 0.5:
            return 'high'
        elif ratio < 0.75:
            return 'medium'
        return 'low'

    def draw(self, image: np.ndarray, detections: Any = None) -> np.ndarray:
        """
        Draw social distancing violations on image.

        Args:
            image: Input image
            detections: Optional YOLO detections for box coloring

        Returns:
            Image with violation overlays
        """
        img_draw = image.copy()

        # Draw person boxes with violation coloring
        if detections is not None and hasattr(detections, 'boxes') and detections.boxes is not None:
            violation_indices = set()
            for v in self.violations:
                violation_indices.add(v['pair'][0])
                violation_indices.add(v['pair'][1])

            for i, box in enumerate(detections.boxes):
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)

                color = (0, 0, 255) if i in violation_indices else (0, 255, 0)
                thickness = 3 if i in violation_indices else 2
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

        # Draw violation lines
        for violation in self.violations:
            p1 = tuple(map(int, violation['centroid1']))
            p2 = tuple(map(int, violation['centroid2']))

            # Color by severity
            severity_colors = {
                'high': (0, 0, 255),
                'medium': (0, 165, 255),
                'low': (0, 255, 255)
            }
            color = severity_colors.get(violation['severity'], (0, 0, 255))

            cv2.line(img_draw, p1, p2, color, 2)

            # Distance label
            mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                img_draw,
                f"{violation['distance']:.0f}px",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Violation count
        cv2.putText(
            img_draw,
            f"Violations: {len(self.violations)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return img_draw

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get social distancing statistics.

        Returns:
            Dictionary with statistics
        """
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for v in self.violations:
            severity_counts[v['severity']] += 1

        return {
            'current_violations': len(self.violations),
            'total_violation_events': sum(self.violation_history.values()),
            'unique_pairs': len(self.violation_history),
            'total_checks': self._total_checks,
            'min_distance_threshold': self.min_distance,
            'severity_breakdown': severity_counts
        }

    def get_violation_pairs(self) -> List[Tuple[int, int]]:
        """Get list of currently violating person pairs."""
        return [v['pair'] for v in self.violations]

    def set_min_distance(self, distance: int) -> None:
        """Update minimum distance threshold."""
        self.min_distance = distance

    def reset(self) -> None:
        """Reset violation tracking."""
        self.violations = []
        self.violation_history.clear()
        self._total_checks = 0
