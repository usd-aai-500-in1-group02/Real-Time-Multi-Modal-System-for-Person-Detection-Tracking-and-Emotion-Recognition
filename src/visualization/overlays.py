"""
Overlay utilities for combining visualizations.
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import cv2

from src.pipeline.orchestrator import ProcessingResult
from src.visualization.drawer import ResultDrawer


class OverlayManager:
    """
    Manage and combine multiple visualization overlays.

    Provides a unified interface for adding various overlays
    to processed images.
    """

    def __init__(self, drawer: Optional[ResultDrawer] = None):
        """
        Initialize the overlay manager.

        Args:
            drawer: ResultDrawer instance (creates new if None)
        """
        self.drawer = drawer or ResultDrawer()
        self._overlays: Dict[str, Callable] = {}
        self._overlay_order: List[str] = []

    def register_overlay(
        self,
        name: str,
        overlay_func: Callable[[np.ndarray], np.ndarray],
        priority: int = 0
    ) -> None:
        """
        Register a custom overlay function.

        Args:
            name: Unique overlay name
            overlay_func: Function(image) -> image
            priority: Lower values are drawn first
        """
        self._overlays[name] = {
            'func': overlay_func,
            'priority': priority,
            'enabled': True
        }
        self._update_order()

    def unregister_overlay(self, name: str) -> None:
        """Remove a registered overlay."""
        if name in self._overlays:
            del self._overlays[name]
            self._update_order()

    def enable_overlay(self, name: str) -> None:
        """Enable an overlay."""
        if name in self._overlays:
            self._overlays[name]['enabled'] = True

    def disable_overlay(self, name: str) -> None:
        """Disable an overlay."""
        if name in self._overlays:
            self._overlays[name]['enabled'] = False

    def _update_order(self) -> None:
        """Update overlay rendering order based on priority."""
        self._overlay_order = sorted(
            self._overlays.keys(),
            key=lambda x: self._overlays[x]['priority']
        )

    def apply_overlays(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all enabled overlays to an image.

        Args:
            image: Input image

        Returns:
            Image with all overlays applied
        """
        result = image.copy()

        for name in self._overlay_order:
            overlay = self._overlays[name]
            if overlay['enabled']:
                result = overlay['func'](result)

        return result

    def create_composite(
        self,
        image: np.ndarray,
        result: ProcessingResult,
        heatmap: Optional[np.ndarray] = None,
        trajectories: Optional[Callable] = None,
        distancing: Optional[Callable] = None,
        flow: Optional[Callable] = None,
        config: Optional[Dict[str, bool]] = None
    ) -> np.ndarray:
        """
        Create a composite image with multiple visualizations.

        Args:
            image: Input image
            result: ProcessingResult
            heatmap: Heatmap overlay function
            trajectories: Trajectory drawer function
            distancing: Social distancing drawer function
            flow: Crowd flow drawer function
            config: Dict of enabled features

        Returns:
            Composite image with all enabled visualizations
        """
        config = config or {}
        img_draw = image.copy()

        # Base result drawing
        img_draw = self.drawer.draw_result(
            img_draw,
            result,
            show_detections=config.get('show_detections', True),
            show_tracks=config.get('show_tracks', True),
            show_masks=config.get('show_masks', False),
            show_emotions=config.get('show_emotions', True),
            show_count=config.get('show_count', True),
            show_fps=config.get('show_fps', True)
        )

        # Additional overlays
        if heatmap is not None and config.get('show_heatmap', False):
            img_draw = heatmap(img_draw)

        if trajectories is not None and config.get('show_trajectories', False):
            img_draw = trajectories(img_draw)

        if distancing is not None and config.get('show_distancing', False):
            img_draw = distancing(img_draw, result.detections)

        if flow is not None and config.get('show_flow', False):
            img_draw = flow(img_draw)

        return img_draw

    @staticmethod
    def add_info_panel(
        image: np.ndarray,
        info: Dict[str, Any],
        position: str = 'top-left',
        bg_alpha: float = 0.7,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Add an information panel to the image.

        Args:
            image: Input image
            info: Dictionary of label: value pairs
            position: Panel position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            bg_alpha: Background transparency
            text_color: Text color
            bg_color: Background color

        Returns:
            Image with info panel
        """
        img_draw = image.copy()
        h, w = image.shape[:2]

        # Calculate panel size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        padding = 10

        lines = [f"{k}: {v}" for k, v in info.items()]
        max_width = max(cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in lines)
        panel_width = max_width + 2 * padding
        panel_height = len(lines) * line_height + 2 * padding

        # Determine position
        if 'left' in position:
            x = padding
        else:
            x = w - panel_width - padding

        if 'top' in position:
            y = padding
        else:
            y = h - panel_height - padding

        # Draw background
        overlay = img_draw.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), bg_color, -1)
        img_draw = cv2.addWeighted(img_draw, 1 - bg_alpha, overlay, bg_alpha, 0)

        # Draw text
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_height - 5
            cv2.putText(img_draw, line, (x + padding, text_y), font, font_scale, text_color, 1)

        return img_draw

    @staticmethod
    def create_grid(
        images: List[np.ndarray],
        grid_size: Tuple[int, int] = None,
        cell_size: Tuple[int, int] = None,
        labels: List[str] = None
    ) -> np.ndarray:
        """
        Create a grid of images.

        Args:
            images: List of images
            grid_size: (rows, cols) or None for auto
            cell_size: (width, height) for each cell
            labels: Optional labels for each image

        Returns:
            Grid image
        """
        n = len(images)
        if n == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Determine grid size
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size

        # Determine cell size
        if cell_size is None:
            cell_w = images[0].shape[1]
            cell_h = images[0].shape[0]
        else:
            cell_w, cell_h = cell_size

        # Create grid
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            # Resize image
            resized = cv2.resize(img, (cell_w, cell_h))

            # Add label if provided
            if labels and i < len(labels):
                cv2.putText(resized, labels[i], (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Place in grid
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            grid[y1:y2, x1:x2] = resized

        return grid

    @staticmethod
    def add_border(
        image: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Add a border around the image."""
        return cv2.copyMakeBorder(
            image,
            thickness, thickness, thickness, thickness,
            cv2.BORDER_CONSTANT,
            value=color
        )

    @staticmethod
    def add_timestamp(
        image: np.ndarray,
        timestamp: str = None,
        position: str = 'bottom-right'
    ) -> np.ndarray:
        """
        Add timestamp to image.

        Args:
            image: Input image
            timestamp: Timestamp string (uses current time if None)
            position: Text position

        Returns:
            Image with timestamp
        """
        from datetime import datetime

        img_draw = image.copy()
        h, w = image.shape[:2]

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), _ = cv2.getTextSize(timestamp, font, font_scale, 1)

        if 'right' in position:
            x = w - text_w - 10
        else:
            x = 10

        if 'bottom' in position:
            y = h - 10
        else:
            y = text_h + 10

        # Background
        cv2.rectangle(img_draw, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(img_draw, timestamp, (x, y), font, font_scale, (255, 255, 255), 1)

        return img_draw
