"""
Trajectory analysis for tracking movement patterns.
Extracted from app4.py lines 480-528.
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import numpy as np
import cv2
import time

from src.analyzers.base import BaseAnalyzer


class TrajectoryAnalyzer(BaseAnalyzer):
    """
    Analyze movement trajectories of tracked persons.

    Tracks position history, calculates speeds, detects loitering,
    and provides trajectory visualizations.
    """

    def __init__(
        self,
        max_history: int = 50,
        loitering_threshold: float = 50.0,
        loitering_min_frames: int = 30
    ):
        """
        Initialize the trajectory analyzer.

        Args:
            max_history: Maximum positions to keep per track
            loitering_threshold: Max movement distance for loitering detection
            loitering_min_frames: Minimum frames for loitering classification
        """
        super().__init__()
        self.max_history = max_history
        self.loitering_threshold = loitering_threshold
        self.loitering_min_frames = loitering_min_frames

        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.track_status: Dict[int, str] = {}
        self.entry_times: Dict[int, float] = {}
        self._is_initialized = True

    def update(self, track_id: int, position: Tuple[float, float]) -> None:
        """
        Update trajectory for a track.

        Args:
            track_id: Unique track identifier
            position: Current position (x, y)
        """
        if track_id not in self.entry_times:
            self.entry_times[track_id] = time.time()

        self.trajectories[track_id].append(position)
        self.track_status[track_id] = 'active'

    def update_from_tracks(self, tracks: List[Any]) -> None:
        """
        Update trajectories from tracker output.

        Args:
            tracks: List of track objects from DeepSort
        """
        # Mark all as inactive first
        for track_id in self.track_status:
            self.track_status[track_id] = 'inactive'

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            self.update(track_id, center)

    def get_dwell_time(self, track_id: int) -> float:
        """
        Get how long a track has been present.

        Args:
            track_id: Track identifier

        Returns:
            Dwell time in seconds
        """
        if track_id in self.entry_times:
            return time.time() - self.entry_times[track_id]
        return 0.0

    def calculate_speed(self, track_id: int, fps: float = 30.0) -> float:
        """
        Calculate average speed of a track.

        Args:
            track_id: Track identifier
            fps: Frames per second

        Returns:
            Average speed in pixels/second
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return 0.0

        trajectory = np.array(list(self.trajectories[track_id]))
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        avg_distance_per_frame = np.mean(distances)

        return avg_distance_per_frame * fps

    def detect_loitering(self, track_id: int) -> bool:
        """
        Detect if a track is loitering.

        Args:
            track_id: Track identifier

        Returns:
            True if loitering detected
        """
        if track_id not in self.trajectories:
            return False

        trajectory = list(self.trajectories[track_id])
        if len(trajectory) < self.loitering_min_frames:
            return False

        trajectory_arr = np.array(trajectory)
        center = np.mean(trajectory_arr, axis=0)
        distances = np.linalg.norm(trajectory_arr - center, axis=1)

        return np.mean(distances) < self.loitering_threshold

    def get_trajectory(self, track_id: int) -> Optional[np.ndarray]:
        """
        Get trajectory positions for a track.

        Args:
            track_id: Track identifier

        Returns:
            Array of positions or None
        """
        if track_id not in self.trajectories:
            return None
        return np.array(list(self.trajectories[track_id]))

    def draw(self, image: np.ndarray) -> np.ndarray:
        """
        Draw trajectories on image.

        Args:
            image: Input image

        Returns:
            Image with trajectory overlays
        """
        img_draw = image.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for i, (track_id, trajectory) in enumerate(self.trajectories.items()):
            if self.track_status.get(track_id) != 'active':
                continue

            color = colors[i % len(colors)]
            trajectory_arr = np.array(list(trajectory), dtype=np.int32)

            if len(trajectory_arr) > 1:
                cv2.polylines(img_draw, [trajectory_arr], False, color, 2)

            if len(trajectory_arr) > 0:
                cv2.circle(img_draw, tuple(trajectory_arr[-1]), 5, color, -1)

                # Show track ID
                cv2.putText(
                    img_draw,
                    f'ID:{track_id}',
                    (trajectory_arr[-1][0] + 10, trajectory_arr[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return img_draw

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trajectory statistics.

        Returns:
            Dictionary with statistics
        """
        active_count = sum(1 for s in self.track_status.values() if s == 'active')
        loitering_count = sum(1 for tid in self.trajectories if self.detect_loitering(tid))

        speeds = [self.calculate_speed(tid) for tid in self.trajectories]
        dwell_times = [self.get_dwell_time(tid) for tid in self.trajectories]

        return {
            'total_tracks': len(self.trajectories),
            'active_tracks': active_count,
            'loitering_count': loitering_count,
            'avg_speed': float(np.mean(speeds)) if speeds else 0.0,
            'max_speed': float(np.max(speeds)) if speeds else 0.0,
            'avg_dwell_time': float(np.mean(dwell_times)) if dwell_times else 0.0,
            'max_dwell_time': float(np.max(dwell_times)) if dwell_times else 0.0
        }

    def get_loitering_tracks(self) -> List[int]:
        """Get list of track IDs that are loitering."""
        return [tid for tid in self.trajectories if self.detect_loitering(tid)]

    def reset(self) -> None:
        """Reset all trajectory data."""
        self.trajectories.clear()
        self.track_status.clear()
        self.entry_times.clear()
