"""
Behavior pattern analysis.
Extracted from app4.py lines 246-317.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np
import time

from src.analyzers.base import BaseAnalyzer


class BehaviorPatternAnalyzer(BaseAnalyzer):
    """
    Detect unusual behavior patterns from tracking data.

    Identifies patterns such as stationary behavior, erratic movement,
    sudden stops, and sudden speed changes.
    """

    PATTERN_TYPES = ['stationary', 'erratic', 'sudden_stop', 'sudden_speed', 'normal']

    def __init__(
        self,
        stationary_threshold: float = 0.1,
        erratic_threshold: float = 5.0
    ):
        """
        Initialize the behavior pattern analyzer.

        Args:
            stationary_threshold: Speed threshold for stationary detection
            erratic_threshold: Variance threshold for erratic movement
        """
        super().__init__()
        self.stationary_threshold = stationary_threshold
        self.erratic_threshold = erratic_threshold

        self.behavior_history: Dict[int, List[Dict]] = defaultdict(list)
        self.anomalies: List[Dict[str, Any]] = []
        self.pattern_counts = {
            'stationary': 0,
            'erratic': 0,
            'sudden_stop': 0,
            'sudden_speed': 0,
            'normal': 0
        }
        self._is_initialized = True

    def update(
        self,
        track_id: int,
        positions: List[tuple]
    ) -> Optional[str]:
        """
        Analyze behavior from position history.

        Args:
            track_id: Track identifier
            positions: List of (x, y) positions

        Returns:
            Detected behavior pattern or None
        """
        return self.analyze_track(track_id, positions)

    def analyze_track(
        self,
        track_id: int,
        positions: List[tuple]
    ) -> Optional[str]:
        """
        Analyze behavior pattern for a track.

        Args:
            track_id: Track identifier
            positions: List of (x, y) positions

        Returns:
            Detected behavior pattern or None
        """
        if len(positions) < 10:
            return None

        positions_arr = np.array(positions)
        velocities = np.diff(positions_arr, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)

        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)

        behavior = None

        # Check for stationary behavior
        if avg_speed < self.stationary_threshold:
            behavior = 'stationary'
            self.pattern_counts['stationary'] += 1

        # Check for erratic movement
        elif speed_variance > self.erratic_threshold:
            behavior = 'erratic'
            self.pattern_counts['erratic'] += 1
            self._add_anomaly(track_id, 'erratic', 'medium')

        # Check for sudden stop
        if len(speeds) > 2 and speeds[-2] > 2.0 and speeds[-1] < 0.5:
            behavior = 'sudden_stop'
            self.pattern_counts['sudden_stop'] += 1

        # Check for sudden speed increase
        if len(speeds) > 2 and speeds[-2] < 1.0 and speeds[-1] > 3.0:
            behavior = 'sudden_speed'
            self.pattern_counts['sudden_speed'] += 1
            self._add_anomaly(track_id, 'sudden_speed', 'high')

        # Record behavior
        if behavior:
            self.behavior_history[track_id].append({
                'behavior': behavior,
                'timestamp': time.time(),
                'avg_speed': avg_speed,
                'speed_variance': speed_variance
            })

        return behavior

    def _add_anomaly(self, track_id: int, behavior: str, severity: str) -> None:
        """Add an anomaly to the list."""
        self.anomalies.append({
            'track_id': track_id,
            'behavior': behavior,
            'severity': severity,
            'timestamp': time.time()
        })

        # Keep only recent anomalies
        if len(self.anomalies) > 100:
            self.anomalies = self.anomalies[-100:]

    def analyze_from_trajectory_analyzer(
        self,
        trajectory_analyzer: Any
    ) -> Dict[int, Optional[str]]:
        """
        Analyze all tracks from a TrajectoryAnalyzer.

        Args:
            trajectory_analyzer: TrajectoryAnalyzer instance

        Returns:
            Dictionary mapping track_id to detected behavior
        """
        results = {}

        for track_id, trajectory in trajectory_analyzer.trajectories.items():
            positions = list(trajectory)
            behavior = self.analyze_track(track_id, positions)
            results[track_id] = behavior

        return results

    def get_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent anomalies.

        Args:
            limit: Maximum number of anomalies to return

        Returns:
            List of recent anomalies
        """
        return self.anomalies[-limit:]

    def get_high_severity_anomalies(self) -> List[Dict[str, Any]]:
        """Get only high severity anomalies."""
        return [a for a in self.anomalies if a['severity'] == 'high']

    def draw(self, image: np.ndarray, positions: Dict[int, tuple] = None) -> np.ndarray:
        """
        Draw behavior indicators on image.

        Args:
            image: Input image
            positions: Dictionary mapping track_id to current position

        Returns:
            Image with behavior annotations
        """
        import cv2
        img_draw = image.copy()

        if positions is None:
            return img_draw

        # Color coding for behaviors
        behavior_colors = {
            'stationary': (0, 255, 255),    # Yellow
            'erratic': (0, 165, 255),       # Orange
            'sudden_stop': (0, 0, 255),     # Red
            'sudden_speed': (255, 0, 0),    # Blue
        }

        for track_id, pos in positions.items():
            if track_id not in self.behavior_history:
                continue

            recent = self.behavior_history[track_id]
            if not recent:
                continue

            latest = recent[-1]
            behavior = latest['behavior']
            color = behavior_colors.get(behavior, (255, 255, 255))

            x, y = int(pos[0]), int(pos[1])

            # Draw indicator circle
            cv2.circle(img_draw, (x, y - 30), 10, color, -1)

            # Draw behavior label
            cv2.putText(
                img_draw,
                behavior[:3].upper(),
                (x - 15, y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        return img_draw

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get behavior analysis statistics.

        Returns:
            Dictionary with statistics
        """
        total_behaviors = sum(self.pattern_counts.values())

        return {
            'total_behaviors_detected': total_behaviors,
            'anomalies_count': len(self.anomalies),
            'high_severity_count': len(self.get_high_severity_anomalies()),
            'pattern_distribution': dict(self.pattern_counts),
            'unique_tracks_analyzed': len(self.behavior_history)
        }

    def get_track_behavior_summary(self, track_id: int) -> Dict[str, Any]:
        """
        Get behavior summary for a specific track.

        Args:
            track_id: Track identifier

        Returns:
            Summary dictionary for the track
        """
        if track_id not in self.behavior_history:
            return {'track_id': track_id, 'behaviors': []}

        history = self.behavior_history[track_id]
        behavior_counts = defaultdict(int)

        for entry in history:
            behavior_counts[entry['behavior']] += 1

        return {
            'track_id': track_id,
            'total_events': len(history),
            'behavior_counts': dict(behavior_counts),
            'recent_behavior': history[-1] if history else None
        }

    def reset(self) -> None:
        """Reset behavior analysis."""
        self.behavior_history.clear()
        self.anomalies.clear()
        self.pattern_counts = {k: 0 for k in self.pattern_counts}
