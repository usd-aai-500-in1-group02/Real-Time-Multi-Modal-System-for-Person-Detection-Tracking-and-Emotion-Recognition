"""
Crowd flow analysis.
Extracted from app4.py lines 113-244.
"""

from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2
import math

from src.analyzers.base import BaseAnalyzer


class CrowdFlowAnalyzer(BaseAnalyzer):
    """
    Analyze crowd movement patterns and flow direction.

    Tracks movement vectors across a spatial grid to visualize
    crowd flow patterns and dominant movement directions.
    """

    DIRECTIONS = [
        'East', 'South-East', 'South', 'South-West',
        'West', 'North-West', 'North', 'North-East'
    ]

    def __init__(self, grid_size: Tuple[int, int] = (4, 4)):
        """
        Initialize the crowd flow analyzer.

        Args:
            grid_size: Grid dimensions for density mapping (rows, cols)
        """
        super().__init__()
        self.grid_size = grid_size
        self.flow_vectors: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.density_map = np.zeros(grid_size, dtype=np.float32)
        self._is_initialized = True

    def update(self, tracks: List[Any], image_shape: Tuple[int, ...]) -> None:
        """
        Analyze crowd flow from tracking data.

        Args:
            tracks: List of track objects from DeepSort
            image_shape: Shape of the image (height, width, ...)
        """
        self.analyze_flow(tracks, image_shape)

    def analyze_flow(self, tracks: List[Any], image_shape: Tuple[int, ...]) -> None:
        """
        Analyze crowd flow patterns.

        Args:
            tracks: List of track objects
            image_shape: Image shape for grid calculation
        """
        height, width = image_shape[:2]
        grid_h, grid_w = self.grid_size
        cell_h, cell_w = height // grid_h, width // grid_w

        # Reset density map
        self.density_map = np.zeros(self.grid_size, dtype=np.float32)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = bbox

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Update grid density
            grid_x = min(int(center_x // cell_w), grid_w - 1)
            grid_y = min(int(center_y // cell_h), grid_h - 1)

            self.density_map[grid_y, grid_x] += 1

            # Update flow vectors
            self.flow_vectors[track_id].append((center_x, center_y))

            # Keep only recent positions
            if len(self.flow_vectors[track_id]) > 30:
                self.flow_vectors[track_id] = self.flow_vectors[track_id][-30:]

    def get_dominant_directions(self) -> Dict[int, Dict[str, Any]]:
        """
        Get dominant movement directions per track.

        Returns:
            Dictionary mapping track_id to direction info
        """
        directions = {}

        for track_id, positions in self.flow_vectors.items():
            if len(positions) < 5:
                continue

            positions_arr = np.array(positions[-10:])
            if len(positions_arr) < 2:
                continue

            # Calculate flow vector
            flow = positions_arr[-1] - positions_arr[0]
            angle = math.atan2(flow[1], flow[0])
            magnitude = np.linalg.norm(flow)

            directions[track_id] = {
                'angle': angle,
                'angle_degrees': math.degrees(angle),
                'magnitude': magnitude,
                'direction': self._angle_to_direction(angle)
            }

        return directions

    def _angle_to_direction(self, angle: float) -> str:
        """
        Convert angle to direction label.

        Args:
            angle: Angle in radians

        Returns:
            Direction string
        """
        angle_deg = math.degrees(angle)

        if -22.5 <= angle_deg < 22.5:
            return 'East'
        elif 22.5 <= angle_deg < 67.5:
            return 'South-East'
        elif 67.5 <= angle_deg < 112.5:
            return 'South'
        elif 112.5 <= angle_deg < 157.5:
            return 'South-West'
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            return 'West'
        elif -157.5 <= angle_deg < -112.5:
            return 'North-West'
        elif -112.5 <= angle_deg < -67.5:
            return 'North'
        else:
            return 'North-East'

    def draw(self, image: np.ndarray) -> np.ndarray:
        """
        Draw flow vectors and density map on image.

        Args:
            image: Input image

        Returns:
            Image with flow visualization
        """
        img_draw = image.copy()
        height, width = image.shape[:2]
        grid_h, grid_w = self.grid_size
        cell_h, cell_w = height // grid_h, width // grid_w

        directions = self.get_dominant_directions()

        # Draw density grid
        for i in range(grid_h):
            for j in range(grid_w):
                if self.density_map[i, j] > 0:
                    density = min(self.density_map[i, j], 5) / 5.0
                    overlay = img_draw.copy()

                    x1, y1 = j * cell_w, i * cell_h
                    x2, y2 = x1 + cell_w, y1 + cell_h

                    # Color: green (low) to red (high)
                    color = (0, int(255 * (1 - density)), int(255 * density))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                    img_draw = cv2.addWeighted(img_draw, 0.7, overlay, 0.3, 0)

        # Draw flow arrows
        for track_id, direction_info in directions.items():
            if track_id not in self.flow_vectors or len(self.flow_vectors[track_id]) == 0:
                continue

            pos = self.flow_vectors[track_id][-1]
            angle = direction_info['angle']
            length = min(direction_info['magnitude'], 50)

            end_x = int(pos[0] + length * math.cos(angle))
            end_y = int(pos[1] + length * math.sin(angle))

            cv2.arrowedLine(
                img_draw,
                (int(pos[0]), int(pos[1])),
                (end_x, end_y),
                (255, 255, 0),
                2,
                tipLength=0.3
            )

        return img_draw

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get crowd flow statistics.

        Returns:
            Dictionary with statistics
        """
        directions = self.get_dominant_directions()
        direction_counts = defaultdict(int)

        for dir_info in directions.values():
            direction_counts[dir_info['direction']] += 1

        return {
            'total_tracked': len(directions),
            'direction_distribution': dict(direction_counts),
            'avg_density': float(np.mean(self.density_map)),
            'max_density': float(np.max(self.density_map)),
            'total_density': float(np.sum(self.density_map)),
            'active_cells': int(np.sum(self.density_map > 0))
        }

    def get_density_map(self) -> np.ndarray:
        """Get the current density map."""
        return self.density_map.copy()

    def get_high_density_zones(self, threshold: float = 3.0) -> List[Tuple[int, int]]:
        """
        Get grid cells with high density.

        Args:
            threshold: Minimum density threshold

        Returns:
            List of (row, col) tuples for high-density cells
        """
        zones = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.density_map[i, j] >= threshold:
                    zones.append((i, j))
        return zones

    def reset(self) -> None:
        """Reset flow analysis."""
        self.flow_vectors.clear()
        self.density_map = np.zeros(self.grid_size, dtype=np.float32)
