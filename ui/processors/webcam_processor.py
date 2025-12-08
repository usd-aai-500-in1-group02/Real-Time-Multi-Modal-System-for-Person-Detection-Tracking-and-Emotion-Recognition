"""
Webcam video processor for WebRTC streaming.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import cv2
import time
from dataclasses import dataclass, field

try:
    from streamlit_webrtc import VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object

from src.pipeline.orchestrator import PipelineOrchestrator, ProcessingConfig, ProcessingResult
from src.visualization.drawer import ResultDrawer


@dataclass
class ProcessingStats:
    """Statistics for webcam processing."""
    frames_processed: int = 0
    total_persons_detected: int = 0
    total_processing_time: float = 0.0
    fps_history: List[float] = field(default_factory=list)
    person_count_history: List[int] = field(default_factory=list)

    def update(self, result: ProcessingResult) -> None:
        """Update stats with new result."""
        self.frames_processed += 1
        self.total_persons_detected += result.person_count
        self.total_processing_time += result.processing_time

        # Keep last 100 entries
        if result.processing_time > 0:
            self.fps_history.append(1.0 / result.processing_time)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)

        self.person_count_history.append(result.person_count)
        if len(self.person_count_history) > 100:
            self.person_count_history.pop(0)

    @property
    def avg_fps(self) -> float:
        """Calculate average FPS."""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

    @property
    def avg_persons(self) -> float:
        """Calculate average person count."""
        return self.total_persons_detected / self.frames_processed if self.frames_processed > 0 else 0.0


class WebcamVideoProcessor(VideoProcessorBase if WEBRTC_AVAILABLE else object):
    """
    Video processor for WebRTC webcam streaming.

    Processes each frame through the detection pipeline and
    returns annotated frames.
    """

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        processing_config: ProcessingConfig
    ):
        """
        Initialize webcam processor.

        Args:
            orchestrator: Pipeline orchestrator for frame processing
            processing_config: Configuration for processing
        """
        self.orchestrator = orchestrator
        self.processing_config = processing_config
        self.drawer = ResultDrawer()
        self.stats = ProcessingStats()

        # Processing state
        self._last_result: Optional[ProcessingResult] = None
        self._frame_count = 0
        self._skip_frames = 0  # Process every frame by default
        self._show_fps = True
        self._show_stats = True

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        """
        Process a video frame from WebRTC.

        Args:
            frame: Input video frame

        Returns:
            Processed video frame with annotations
        """
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Frame skipping for performance
        self._frame_count += 1
        if self._skip_frames > 0 and self._frame_count % (self._skip_frames + 1) != 0:
            # Return previous result visualization if skipping
            if self._last_result is not None:
                output_img = self._draw_cached_result(img)
            else:
                output_img = img
        else:
            # Process frame
            output_img = self._process_frame(img)

        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(output_img, format="bgr24")

    def _process_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the pipeline.

        Args:
            img: Input image in BGR format

        Returns:
            Annotated image
        """
        # Run detection pipeline
        result = self.orchestrator.process_frame(img, self.processing_config)
        self._last_result = result

        # Update statistics
        self.stats.update(result)

        # Draw results
        output_img = self.drawer.draw_full_results(
            img.copy(),
            result,
            show_emotions=self.processing_config.emotion,
            show_masks=self.processing_config.segmentation
        )

        # Add overlays
        if self._show_fps:
            output_img = self._draw_fps_overlay(output_img, result)

        if self._show_stats:
            output_img = self._draw_stats_overlay(output_img)

        return output_img

    def _draw_cached_result(self, img: np.ndarray) -> np.ndarray:
        """Draw cached result on new frame (for frame skipping)."""
        if self._last_result is None:
            return img

        output_img = self.drawer.draw_full_results(
            img.copy(),
            self._last_result,
            show_emotions=self.processing_config.emotion,
            show_masks=self.processing_config.segmentation
        )

        if self._show_fps:
            # Show "cached" indicator
            cv2.putText(
                output_img,
                "CACHED",
                (img.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 165, 0),
                2
            )

        return output_img

    def _draw_fps_overlay(self, img: np.ndarray, result: ProcessingResult) -> np.ndarray:
        """Draw FPS and processing time overlay."""
        fps = 1.0 / result.processing_time if result.processing_time > 0 else 0

        # Background rectangle
        cv2.rectangle(img, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (200, 80), (100, 100, 100), 1)

        # FPS text
        fps_color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 5 else (0, 0, 255)
        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            fps_color,
            2
        )

        # Processing time
        cv2.putText(
            img,
            f"Time: {result.processing_time*1000:.0f}ms",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return img

    def _draw_stats_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw statistics overlay."""
        h, w = img.shape[:2]

        # Background rectangle at bottom
        cv2.rectangle(img, (10, h - 60), (250, h - 10), (0, 0, 0), -1)
        cv2.rectangle(img, (10, h - 60), (250, h - 10), (100, 100, 100), 1)

        # Stats text
        cv2.putText(
            img,
            f"Frames: {self.stats.frames_processed}",
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img,
            f"Avg Persons: {self.stats.avg_persons:.1f}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return img

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'frames_processed': self.stats.frames_processed,
            'total_persons': self.stats.total_persons_detected,
            'avg_fps': self.stats.avg_fps,
            'avg_persons': self.stats.avg_persons,
            'current_persons': self._last_result.person_count if self._last_result else 0,
        }

    def set_skip_frames(self, skip: int) -> None:
        """Set frame skipping for performance tuning."""
        self._skip_frames = max(0, skip)

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = ProcessingStats()
        self._frame_count = 0


class SimpleWebcamProcessor:
    """
    Simple webcam processor for non-WebRTC environments.

    Uses OpenCV for webcam capture and processing.
    """

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        processing_config: ProcessingConfig,
        camera_id: int = 0
    ):
        """
        Initialize simple webcam processor.

        Args:
            orchestrator: Pipeline orchestrator
            processing_config: Processing configuration
            camera_id: Camera device ID (default 0)
        """
        self.orchestrator = orchestrator
        self.processing_config = processing_config
        self.camera_id = camera_id
        self.drawer = ResultDrawer()
        self.stats = ProcessingStats()

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False

    def start(self) -> bool:
        """Start webcam capture."""
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            return False
        self._running = True
        return True

    def stop(self) -> None:
        """Stop webcam capture."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from webcam."""
        if self._cap is None or not self._running:
            return None

        ret, frame = self._cap.read()
        return frame if ret else None

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame.

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (annotated_frame, processing_result)
        """
        # Run detection pipeline
        result = self.orchestrator.process_frame(frame, self.processing_config)
        self.stats.update(result)

        # Draw results
        output_frame = self.drawer.draw_full_results(
            frame.copy(),
            result,
            show_emotions=self.processing_config.emotion,
            show_masks=self.processing_config.segmentation
        )

        return output_frame, result

    @property
    def is_running(self) -> bool:
        """Check if webcam is running."""
        return self._running and self._cap is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'frames_processed': self.stats.frames_processed,
            'avg_fps': self.stats.avg_fps,
            'avg_persons': self.stats.avg_persons,
        }
