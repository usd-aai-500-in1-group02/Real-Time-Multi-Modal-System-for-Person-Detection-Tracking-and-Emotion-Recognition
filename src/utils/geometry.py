"""
Geometry utilities for bounding box and spatial calculations.
"""

from typing import Tuple, List, Union
import numpy as np
import cv2


def calculate_centroid(bbox: Union[List[float], Tuple[float, ...], np.ndarray]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_bottom_centroid(bbox: Union[List[float], Tuple[float, ...], np.ndarray]) -> Tuple[float, float]:
    """
    Calculate the bottom center point of a bounding box.
    Useful for ground-plane distance calculations.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Tuple of (center_x, bottom_y)
    """
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2, y2)


def euclidean_distance(
    point1: Union[Tuple[float, float], np.ndarray],
    point2: Union[Tuple[float, float], np.ndarray]
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Euclidean distance as float
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return float(np.linalg.norm(p1 - p2))


def bbox_to_xywh(bbox: Union[List[float], Tuple[float, ...], np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x1, y1, x2, y2] to [x, y, width, height] format.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Tuple of (x, y, width, height)
    """
    x1, y1, x2, y2 = bbox[:4]
    return (x1, y1, x2 - x1, y2 - y1)


def bbox_to_xyxy(bbox: Union[List[float], Tuple[float, ...], np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x, y, width, height] to [x1, y1, x2, y2] format.

    Args:
        bbox: Bounding box in [x, y, width, height] format

    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    x, y, w, h = bbox[:4]
    return (x, y, x + w, y + h)


def is_point_in_polygon(
    point: Union[Tuple[float, float], List[float]],
    polygon: np.ndarray
) -> bool:
    """
    Check if a point is inside a polygon.

    Args:
        point: Point as (x, y)
        polygon: Polygon vertices as numpy array of shape (N, 2)

    Returns:
        True if point is inside polygon
    """
    result = cv2.pointPolygonTest(polygon.astype(np.int32), tuple(map(int, point)), False)
    return result >= 0


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def get_bbox_area(bbox: Union[List[float], Tuple[float, ...], np.ndarray]) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Area in pixels squared
    """
    x1, y1, x2, y2 = bbox[:4]
    return abs((x2 - x1) * (y2 - y1))


def scale_bbox(
    bbox: Union[List[float], Tuple[float, ...], np.ndarray],
    scale: float,
    image_shape: Tuple[int, int] = None
) -> Tuple[float, float, float, float]:
    """
    Scale a bounding box around its center.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        scale: Scale factor (1.0 = no change, 1.5 = 50% larger)
        image_shape: Optional (height, width) to clip to image bounds

    Returns:
        Scaled bounding box as tuple
    """
    x1, y1, x2, y2 = bbox[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    new_w, new_h = w * scale, h * scale

    new_x1 = cx - new_w / 2
    new_y1 = cy - new_h / 2
    new_x2 = cx + new_w / 2
    new_y2 = cy + new_h / 2

    if image_shape is not None:
        img_h, img_w = image_shape
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_w, new_x2)
        new_y2 = min(img_h, new_y2)

    return (new_x1, new_y1, new_x2, new_y2)
