"""
Queue management system.
Extracted from app4.py lines 319-402.
"""

from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2
import time

from src.analyzers.base import BaseAnalyzer


class QueueManagementSystem(BaseAnalyzer):
    """
    Monitor and manage queues in defined zones.

    Tracks persons in queue areas, calculates wait times,
    and provides queue analytics.
    """

    def __init__(self):
        """Initialize the queue management system."""
        super().__init__()
        self.queues: Dict[str, Dict[str, Any]] = {}
        self.wait_times: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.queue_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'current_count': 0,
                'avg_wait_time': 0.0,
                'max_length': 0,
                'total_served': 0,
                'persons': []
            }
        )
        self._is_initialized = True

    def add_queue(
        self,
        name: str,
        polygon: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (255, 165, 0)
    ) -> None:
        """
        Add a queue monitoring zone.

        Args:
            name: Queue name identifier
            polygon: List of polygon vertices
            color: BGR color for visualization
        """
        self.queues[name] = {
            'polygon': np.array(polygon, dtype=np.int32),
            'color': color,
            'persons': []
        }

    def remove_queue(self, name: str) -> bool:
        """
        Remove a queue zone.

        Args:
            name: Queue name

        Returns:
            True if removed successfully
        """
        if name in self.queues:
            del self.queues[name]
            return True
        return False

    def update(self, tracks: List[Any]) -> None:
        """
        Update queue status from tracks.

        Args:
            tracks: List of track objects from DeepSort
        """
        self.update_queues(tracks)

    def update_queues(self, tracks: List[Any]) -> None:
        """
        Update all queues with current tracks.

        Args:
            tracks: List of track objects
        """
        current_time = time.time()

        for queue_name, queue_data in self.queues.items():
            previous_persons = set(queue_data['persons'])
            queue_data['persons'] = []

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                # Check if person is in queue zone
                is_in_queue = cv2.pointPolygonTest(
                    queue_data['polygon'],
                    tuple(map(int, center)),
                    False
                ) >= 0

                if is_in_queue:
                    queue_data['persons'].append(track_id)

                    # Record entry time if new
                    if track_id not in self.wait_times[queue_name]:
                        self.wait_times[queue_name][track_id] = current_time

            # Check for persons who left the queue
            current_persons = set(queue_data['persons'])
            left_queue = previous_persons - current_persons

            for track_id in left_queue:
                if track_id in self.wait_times[queue_name]:
                    del self.wait_times[queue_name][track_id]
                    self.queue_stats[queue_name]['total_served'] += 1

            # Update statistics
            current_count = len(queue_data['persons'])
            self.queue_stats[queue_name]['current_count'] = current_count
            self.queue_stats[queue_name]['max_length'] = max(
                self.queue_stats[queue_name]['max_length'],
                current_count
            )
            self.queue_stats[queue_name]['persons'] = queue_data['persons'].copy()

            # Calculate average wait time
            if self.wait_times[queue_name]:
                wait_times_list = [
                    current_time - entry_time
                    for entry_time in self.wait_times[queue_name].values()
                ]
                self.queue_stats[queue_name]['avg_wait_time'] = np.mean(wait_times_list)
            else:
                self.queue_stats[queue_name]['avg_wait_time'] = 0.0

    def draw(self, image: np.ndarray) -> np.ndarray:
        """
        Draw queue zones and statistics on image.

        Args:
            image: Input image

        Returns:
            Image with queue overlays
        """
        img_draw = image.copy()

        for queue_name, queue_data in self.queues.items():
            color = queue_data['color']

            # Draw polygon
            cv2.polylines(img_draw, [queue_data['polygon']], True, color, 3)

            # Draw fill with transparency
            overlay = img_draw.copy()
            cv2.fillPoly(overlay, [queue_data['polygon']], color)
            img_draw = cv2.addWeighted(img_draw, 0.9, overlay, 0.1, 0)

            # Draw statistics
            center = np.mean(queue_data['polygon'], axis=0).astype(int)
            stats = self.queue_stats[queue_name]

            info_text = f"{queue_name}: {stats['current_count']} persons"
            cv2.putText(
                img_draw,
                info_text,
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            wait_text = f"Avg Wait: {stats['avg_wait_time']:.1f}s"
            cv2.putText(
                img_draw,
                wait_text,
                (center[0], center[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return img_draw

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics for all queues.

        Returns:
            Dictionary with statistics per queue
        """
        return {name: dict(stats) for name, stats in self.queue_stats.items()}

    def get_queue_info(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific queue.

        Args:
            queue_name: Queue name

        Returns:
            Queue information or None
        """
        if queue_name not in self.queues:
            return None

        return {
            'name': queue_name,
            'polygon': self.queues[queue_name]['polygon'].tolist(),
            'color': self.queues[queue_name]['color'],
            **self.queue_stats[queue_name]
        }

    def get_total_persons_in_queues(self) -> int:
        """Get total number of persons across all queues."""
        return sum(stats['current_count'] for stats in self.queue_stats.values())

    def get_busiest_queue(self) -> Optional[str]:
        """Get the name of the busiest queue."""
        if not self.queue_stats:
            return None

        return max(
            self.queue_stats.items(),
            key=lambda x: x[1]['current_count']
        )[0]

    def get_wait_time(self, queue_name: str, track_id: int) -> float:
        """
        Get wait time for a specific person in a queue.

        Args:
            queue_name: Queue name
            track_id: Track identifier

        Returns:
            Wait time in seconds
        """
        if queue_name not in self.wait_times:
            return 0.0

        if track_id not in self.wait_times[queue_name]:
            return 0.0

        return time.time() - self.wait_times[queue_name][track_id]

    def reset(self) -> None:
        """Reset queue statistics."""
        self.wait_times.clear()
        for name in self.queue_stats:
            self.queue_stats[name] = {
                'current_count': 0,
                'avg_wait_time': 0.0,
                'max_length': 0,
                'total_served': 0,
                'persons': []
            }
        for queue_data in self.queues.values():
            queue_data['persons'] = []
