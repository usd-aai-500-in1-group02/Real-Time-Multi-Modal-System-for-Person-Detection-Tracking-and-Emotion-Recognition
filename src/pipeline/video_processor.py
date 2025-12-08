"""
Video file processor for frame-by-frame processing.
NEW: Implements missing PDF requirement for video file processing with tracking.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable, Generator
from pathlib import Path
import time
import cv2
import numpy as np

from src.pipeline.orchestrator import PipelineOrchestrator, ProcessingConfig, ProcessingResult


@dataclass
class VideoInfo:
    """Information about a video file."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    codec: str


@dataclass
class VideoProcessingResult:
    """Results from processing an entire video."""
    video_info: VideoInfo
    frame_results: List[ProcessingResult]
    total_processing_time: float
    avg_fps: float
    output_path: Optional[str] = None


class VideoFileProcessor:
    """
    Process video files frame-by-frame with full pipeline.

    Implements the missing PDF requirement for video file
    processing with tracking persistence across frames.
    """

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        output_codec: str = "mp4v"
    ):
        """
        Initialize the video processor.

        Args:
            orchestrator: PipelineOrchestrator instance
            output_codec: FourCC codec for output video
        """
        self.orchestrator = orchestrator
        self.output_codec = output_codec

    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Get information about a video file.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo object
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=total_frames / fps if fps > 0 else 0,
            codec=codec
        )

    def process_video(
        self,
        video_path: str,
        config: Optional[ProcessingConfig] = None,
        output_path: Optional[str] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, ProcessingResult], None]] = None,
        draw_callback: Optional[Callable[[np.ndarray, ProcessingResult], np.ndarray]] = None
    ) -> VideoProcessingResult:
        """
        Process an entire video file.

        Args:
            video_path: Path to input video
            config: Processing configuration
            output_path: Optional path for output video with visualizations
            skip_frames: Process every Nth frame (0 = no skip)
            max_frames: Maximum frames to process (None = all)
            progress_callback: Callback(current_frame, total_frames, result)
            draw_callback: Callback(frame, result) -> annotated_frame

        Returns:
            VideoProcessingResult with all frame results
        """
        config = config or ProcessingConfig()
        video_info = self.get_video_info(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                video_info.fps,
                (video_info.width, video_info.height)
            )

        # Reset orchestrator for fresh processing
        self.orchestrator.reset_all()

        frame_results: List[ProcessingResult] = []
        start_time = time.time()
        frame_idx = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check max frames
                if max_frames and processed_count >= max_frames:
                    break

                # Skip frames if configured
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Process frame
                result = self.orchestrator.process_frame(
                    frame,
                    config=config,
                    frame_num=frame_idx
                )
                frame_results.append(result)
                processed_count += 1

                # Draw and write output
                if writer and draw_callback:
                    annotated_frame = draw_callback(frame, result)
                    writer.write(annotated_frame)
                elif writer:
                    writer.write(frame)

                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx, video_info.total_frames, result, frame)

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()

        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0

        return VideoProcessingResult(
            video_info=video_info,
            frame_results=frame_results,
            total_processing_time=total_time,
            avg_fps=avg_fps,
            output_path=output_path
        )

    def process_video_generator(
        self,
        video_path: str,
        config: Optional[ProcessingConfig] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None
    ) -> Generator[tuple, None, None]:
        """
        Process video as a generator (for streaming/real-time display).

        Args:
            video_path: Path to input video
            config: Processing configuration
            skip_frames: Process every Nth frame
            max_frames: Maximum frames to process

        Yields:
            Tuple of (frame_number, frame, result)
        """
        config = config or ProcessingConfig()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.orchestrator.reset_all()
        frame_idx = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and processed_count >= max_frames:
                    break

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                result = self.orchestrator.process_frame(
                    frame,
                    config=config,
                    frame_num=frame_idx
                )

                yield (frame_idx, frame, result)

                processed_count += 1
                frame_idx += 1

        finally:
            cap.release()

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: int = 30,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames from video at regular intervals.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval: Extract every Nth frame
            max_frames: Maximum frames to extract

        Returns:
            List of saved frame paths
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        saved_paths = []
        frame_idx = 0
        saved_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and saved_count >= max_frames:
                    break

                if frame_idx % interval == 0:
                    frame_path = output_path / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    saved_paths.append(str(frame_path))
                    saved_count += 1

                frame_idx += 1

        finally:
            cap.release()

        return saved_paths

    def create_summary_video(
        self,
        video_path: str,
        output_path: str,
        config: Optional[ProcessingConfig] = None,
        draw_callback: Optional[Callable[[np.ndarray, ProcessingResult], np.ndarray]] = None,
        target_fps: Optional[float] = None
    ) -> VideoProcessingResult:
        """
        Create a summary video with all visualizations.

        Args:
            video_path: Input video path
            output_path: Output video path
            config: Processing configuration
            draw_callback: Function to draw visualizations
            target_fps: Output FPS (None = same as input)

        Returns:
            VideoProcessingResult
        """
        config = config or ProcessingConfig(
            detection=True,
            tracking=True,
            enable_trajectories=True
        )

        return self.process_video(
            video_path=video_path,
            config=config,
            output_path=output_path,
            draw_callback=draw_callback
        )

    def get_processing_statistics(
        self,
        result: VideoProcessingResult
    ) -> Dict[str, Any]:
        """
        Get statistics from video processing result.

        Args:
            result: VideoProcessingResult

        Returns:
            Statistics dictionary
        """
        if not result.frame_results:
            return {}

        person_counts = [r.person_count for r in result.frame_results]
        processing_times = [r.processing_time for r in result.frame_results]
        violations = [r.distancing_violations for r in result.frame_results]

        return {
            'video_info': {
                'duration': result.video_info.duration,
                'total_frames': result.video_info.total_frames,
                'fps': result.video_info.fps,
                'resolution': f"{result.video_info.width}x{result.video_info.height}"
            },
            'processing': {
                'frames_processed': len(result.frame_results),
                'total_time': result.total_processing_time,
                'avg_fps': result.avg_fps
            },
            'detection': {
                'avg_persons': float(np.mean(person_counts)),
                'max_persons': int(max(person_counts)),
                'total_detections': sum(person_counts)
            },
            'performance': {
                'avg_processing_time_ms': float(np.mean(processing_times) * 1000),
                'min_processing_time_ms': float(min(processing_times) * 1000),
                'max_processing_time_ms': float(max(processing_times) * 1000)
            },
            'violations': {
                'total': sum(violations),
                'frames_with_violations': sum(1 for v in violations if v > 0)
            }
        }
