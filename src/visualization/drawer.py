"""
Result drawer for visualizing detection and analysis results.
Refactored from draw_results() in app4.py lines 819-861.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2

from src.pipeline.orchestrator import ProcessingResult


class ResultDrawer:
    """
    Draw detection and analysis results on images.

    Provides methods to visualize bounding boxes, segmentation masks,
    track IDs, emotions, and other analysis outputs.
    """

    # Default colors (BGR format)
    COLORS = {
        'detection': (0, 255, 0),      # Green
        'track': (255, 255, 0),        # Cyan
        'violation': (0, 0, 255),       # Red
        'safe': (0, 255, 0),           # Green
        'emotion': (0, 255, 255),      # Yellow
        'face': (255, 0, 255),         # Magenta
        'mask': (255, 0, 255),         # Magenta
        'text_bg': (0, 0, 0),          # Black
    }

    EMOTION_COLORS = {
        'happy': (0, 255, 0),
        'sad': (255, 0, 0),
        'angry': (0, 0, 255),
        'neutral': (128, 128, 128),
        'surprise': (0, 255, 255),
        'fear': (255, 0, 255),
        'disgust': (0, 128, 128),
    }

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        thickness: int = 2
    ):
        """
        Initialize the result drawer.

        Args:
            font: OpenCV font type
            font_scale: Font scale factor
            thickness: Line thickness
        """
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Any,
        color: Tuple[int, int, int] = None,
        show_confidence: bool = True,
        show_index: bool = True
    ) -> np.ndarray:
        """
        Draw detection bounding boxes.

        Args:
            image: Input image
            detections: YOLO detection results
            color: Box color (BGR)
            show_confidence: Show confidence score
            show_index: Show person index

        Returns:
            Annotated image
        """
        img_draw = image.copy()
        color = color or self.COLORS['detection']

        if detections is None or not hasattr(detections, 'boxes') or detections.boxes is None:
            return img_draw

        for i, box in enumerate(detections.boxes):
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, bbox)

            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, self.thickness)

            # Label
            label_parts = []
            if show_index:
                label_parts.append(f'Person {i+1}')
            if show_confidence:
                label_parts.append(f'{conf:.2f}')

            if label_parts:
                label = ': '.join(label_parts)
                self._draw_label(img_draw, label, (x1, y1 - 10), color)

        return img_draw

    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Any],
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Draw track IDs on detections.

        Args:
            image: Input image
            tracks: List of track objects
            color: Text color (BGR)

        Returns:
            Annotated image
        """
        img_draw = image.copy()
        color = color or self.COLORS['track']

        if not tracks:
            return img_draw

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)

            # Draw ID label
            label = f'ID: {track_id}'
            self._draw_label(img_draw, label, (x1, y2 + 20), color)

        return img_draw

    def draw_segmentation_masks(
        self,
        image: np.ndarray,
        segmentation: Any,
        color: Tuple[int, int, int] = None,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw segmentation masks overlay.

        Args:
            image: Input image
            segmentation: YOLO segmentation results
            color: Mask color (BGR)
            alpha: Mask transparency

        Returns:
            Image with mask overlay
        """
        img_draw = image.copy()
        color = color or self.COLORS['mask']

        if segmentation is None or not hasattr(segmentation, 'masks') or segmentation.masks is None:
            return img_draw

        masks = segmentation.masks.data.cpu().numpy()

        for mask in masks:
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            colored_mask = np.zeros_like(image)
            colored_mask[:, :] = color

            mask_3ch = np.stack([mask_resized, mask_resized, mask_resized], axis=-1)
            img_draw = np.where(
                mask_3ch > 0.5,
                cv2.addWeighted(img_draw, 1 - alpha, colored_mask, alpha, 0),
                img_draw
            )

        return img_draw

    def draw_emotions(
        self,
        image: np.ndarray,
        detections: Any,
        emotions: List[Optional[Dict]],
        use_emotion_colors: bool = True
    ) -> np.ndarray:
        """
        Draw emotion labels on detected persons.

        Args:
            image: Input image
            detections: YOLO detection results
            emotions: List of emotion dictionaries
            use_emotion_colors: Use emotion-specific colors

        Returns:
            Annotated image
        """
        img_draw = image.copy()

        if detections is None or not hasattr(detections, 'boxes') or detections.boxes is None:
            return img_draw

        for i, emotion_data in enumerate(emotions):
            if emotion_data is None or i >= len(detections.boxes):
                continue

            bbox = detections.boxes[i].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)

            emotion = emotion_data.get('emotion', 'unknown')
            confidence = emotion_data.get('confidence', 0)

            if use_emotion_colors:
                color = self.EMOTION_COLORS.get(emotion, self.COLORS['emotion'])
            else:
                color = self.COLORS['emotion']

            label = f"{emotion}: {confidence:.2f}"
            self._draw_label(img_draw, label, (x1, y2 + 40), color)

        return img_draw

    def draw_faces(
        self,
        image: np.ndarray,
        faces: List[Dict],
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Draw face detection boxes.

        Args:
            image: Input image
            faces: List of face detection dictionaries
            color: Box color

        Returns:
            Annotated image
        """
        img_draw = image.copy()
        color = color or self.COLORS['face']

        for face in faces:
            bbox = face.get('bbox', face.get('box', None))
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 1)

            # Draw landmarks if available
            landmarks = face.get('landmarks', face.get('keypoints', {}))
            if landmarks:
                for point_name, point in landmarks.items():
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        cv2.circle(img_draw, (int(point[0]), int(point[1])), 2, color, -1)

        return img_draw

    def draw_count(
        self,
        image: np.ndarray,
        count: int,
        position: Tuple[int, int] = (10, 30),
        prefix: str = "Count:"
    ) -> np.ndarray:
        """
        Draw person count on image.

        Args:
            image: Input image
            count: Number to display
            position: Text position (x, y)
            prefix: Label prefix

        Returns:
            Annotated image
        """
        img_draw = image.copy()
        label = f"{prefix} {count}"

        # Draw background
        (w, h), _ = cv2.getTextSize(label, self.font, 1.0, 2)
        cv2.rectangle(img_draw, (position[0] - 5, position[1] - h - 5),
                     (position[0] + w + 5, position[1] + 5), (0, 0, 0), -1)

        cv2.putText(img_draw, label, position, self.font, 1.0, (0, 255, 0), 2)

        return img_draw

    def draw_fps(
        self,
        image: np.ndarray,
        fps: float,
        position: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Draw FPS indicator.

        Args:
            image: Input image
            fps: Frames per second
            position: Text position

        Returns:
            Annotated image
        """
        img_draw = image.copy()

        if position is None:
            position = (image.shape[1] - 120, 30)

        label = f"FPS: {fps:.1f}"
        color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)

        cv2.putText(img_draw, label, position, self.font, 0.7, color, 2)

        return img_draw

    def draw_social_distancing_violations(
        self,
        image: np.ndarray,
        detections: Any,
        min_distance: float = 150.0
    ) -> np.ndarray:
        """
        Draw social distancing violation lines.

        Args:
            image: Input image
            detections: YOLO detection results
            min_distance: Minimum safe distance in pixels

        Returns:
            Annotated image
        """
        img_draw = image.copy()

        if detections is None or not hasattr(detections, 'boxes') or detections.boxes is None:
            return img_draw

        boxes = detections.boxes
        if len(boxes) < 2:
            return img_draw

        # Calculate centroids
        centroids = []
        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))

        # Check distances and draw violations
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 +
                              (centroids[i][1] - centroids[j][1])**2)

                if dist < min_distance:
                    # Draw red line for violation
                    cv2.line(img_draw, centroids[i], centroids[j], (0, 0, 255), 2)
                    # Draw distance text at midpoint
                    mid_x = (centroids[i][0] + centroids[j][0]) // 2
                    mid_y = (centroids[i][1] + centroids[j][1]) // 2
                    cv2.putText(img_draw, f"{int(dist)}px", (mid_x, mid_y),
                               self.font, 0.4, (0, 0, 255), 1)

        return img_draw

    def draw_result(
        self,
        image: np.ndarray,
        result: ProcessingResult,
        show_detections: bool = True,
        show_tracks: bool = True,
        show_masks: bool = False,
        show_emotions: bool = True,
        show_count: bool = True,
        show_fps: bool = True,
        show_distancing: bool = False,
        min_distance: float = 150.0
    ) -> np.ndarray:
        """
        Draw all results from a ProcessingResult.

        Args:
            image: Input image
            result: ProcessingResult object
            show_detections: Draw detection boxes
            show_tracks: Draw track IDs
            show_masks: Draw segmentation masks
            show_emotions: Draw emotion labels
            show_count: Draw person count
            show_fps: Draw FPS indicator
            show_distancing: Draw social distancing violations
            min_distance: Minimum safe distance for social distancing

        Returns:
            Fully annotated image
        """
        img_draw = image.copy()

        # Segmentation masks (draw first, below other annotations)
        if show_masks and result.segmentation is not None:
            img_draw = self.draw_segmentation_masks(img_draw, result.segmentation)

        # Social distancing violations (before boxes so lines are behind)
        if show_distancing and result.detections is not None:
            img_draw = self.draw_social_distancing_violations(img_draw, result.detections, min_distance)

        # Detections
        if show_detections and result.detections is not None:
            img_draw = self.draw_detections(img_draw, result.detections)

        # Tracks
        if show_tracks and result.tracks:
            img_draw = self.draw_tracks(img_draw, result.tracks)

        # Emotions
        if show_emotions and result.emotions and result.detections is not None:
            img_draw = self.draw_emotions(img_draw, result.detections, result.emotions)

        # Count
        if show_count:
            img_draw = self.draw_count(img_draw, result.person_count)

        # FPS
        if show_fps and result.processing_time > 0:
            fps = 1.0 / result.processing_time
            img_draw = self.draw_fps(img_draw, fps)

        return img_draw

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = None
    ) -> None:
        """
        Draw text label with optional background.

        Args:
            image: Image to draw on (modified in place)
            text: Label text
            position: Text position (x, y)
            color: Text color
            bg_color: Background color (None for no background)
        """
        x, y = position

        if bg_color is not None:
            (w, h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
            cv2.rectangle(image, (x, y - h - 2), (x + w, y + 2), bg_color, -1)

        cv2.putText(image, text, (x, y), self.font, self.font_scale, color, self.thickness)

    # Alias for compatibility
    def draw_full_results(
        self,
        image: np.ndarray,
        result: ProcessingResult,
        show_emotions: bool = True,
        show_masks: bool = False,
        show_distancing: bool = False,
        min_distance: float = 150.0
    ) -> np.ndarray:
        """
        Draw all results (alias for draw_result).

        Args:
            image: Input image
            result: ProcessingResult object
            show_emotions: Draw emotion labels
            show_masks: Draw segmentation masks
            show_distancing: Draw social distancing violations
            min_distance: Minimum safe distance for social distancing

        Returns:
            Fully annotated image
        """
        return self.draw_result(
            image,
            result,
            show_detections=True,
            show_tracks=True,
            show_masks=show_masks,
            show_emotions=show_emotions,
            show_count=True,
            show_fps=True,
            show_distancing=show_distancing,
            min_distance=min_distance
        )
