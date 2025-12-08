"""
Heatmap generation for density visualization.
Extracted from app4.py lines 455-478.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from src.analyzers.base import BaseAnalyzer


class HeatmapGenerator(BaseAnalyzer):
    """
    Generate density heatmaps from person detections.

    Accumulates detection locations over time to create
    a visualization of where people spend the most time.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int, int],
        sigma: float = 50.0,
        decay: float = 0.0
    ):
        """
        Initialize the heatmap generator.

        Args:
            frame_shape: Shape of video frames (height, width, channels)
            sigma: Standard deviation for Gaussian kernel
            decay: Decay factor per frame (0 = no decay)
        """
        super().__init__()
        self.height, self.width = frame_shape[:2]
        self.sigma = sigma
        self.decay = decay
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self._detection_count = 0
        self._is_initialized = True

    def update(self, bbox: Tuple[float, float, float, float], weight: float = 1.0) -> None:
        """
        Add a detection to the heatmap.

        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            weight: Weight for this detection
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Create Gaussian blob
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        gaussian = np.exp(
            -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * self.sigma ** 2)
        )

        self.heatmap += gaussian * weight
        self._detection_count += 1

        # Apply decay if configured
        if self.decay > 0:
            self.heatmap *= (1 - self.decay)

    def update_from_detections(self, detections: Any) -> None:
        """
        Update heatmap from YOLO detection results.

        Args:
            detections: YOLO detection results object
        """
        if detections is None or not hasattr(detections, 'boxes') or detections.boxes is None:
            return

        for box in detections.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            self.update(bbox, weight=conf)

    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Get the current heatmap.

        Args:
            normalize: Whether to normalize to 0-255 range

        Returns:
            Heatmap as uint8 array
        """
        if normalize and self.heatmap.max() > 0:
            normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
            return normalized
        return self.heatmap.astype(np.uint8)

    def get_colored_heatmap(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Get colored heatmap using OpenCV colormap.

        Args:
            colormap: OpenCV colormap constant

        Returns:
            BGR colored heatmap
        """
        heatmap = self.get_heatmap(normalize=True)
        return cv2.applyColorMap(heatmap, colormap)

    def draw(self, image: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Overlay heatmap on image.

        Args:
            image: Input image
            alpha: Transparency of heatmap overlay

        Returns:
            Image with heatmap overlay
        """
        heatmap_colored = self.get_colored_heatmap()

        # Resize heatmap to match input image size if needed
        if heatmap_colored.shape[:2] != image.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get heatmap statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'detection_count': self._detection_count,
            'max_density': float(self.heatmap.max()),
            'mean_density': float(self.heatmap.mean()),
            'total_density': float(self.heatmap.sum()),
            'hot_spots': self._find_hot_spots()
        }

    def _find_hot_spots(self, threshold: float = 0.8, max_spots: int = 5) -> list:
        """Find high-density hot spots."""
        if self.heatmap.max() == 0:
            return []

        normalized = self.heatmap / self.heatmap.max()
        hot_mask = normalized > threshold

        # Find contours of hot areas
        hot_uint8 = (hot_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(hot_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots = []
        for contour in contours[:max_spots]:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                area = cv2.contourArea(contour)
                spots.append({'center': (cx, cy), 'area': area})

        return spots

    def reset(self) -> None:
        """Reset the heatmap."""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self._detection_count = 0
