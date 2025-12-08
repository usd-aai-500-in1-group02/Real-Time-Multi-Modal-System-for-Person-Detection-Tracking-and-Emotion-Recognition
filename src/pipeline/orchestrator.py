"""
Pipeline orchestrator for coordinating the full processing pipeline.
Refactored from process_image_enhanced() in app4.py lines 865-927.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import time
import numpy as np

from src.config.settings import AppConfig
from src.models.loader import ModelLoader
from src.analyzers.heatmap import HeatmapGenerator
from src.analyzers.trajectory import TrajectoryAnalyzer
from src.analyzers.roi import ROIManager
from src.advanced.social_distancing import SocialDistancingMonitor
from src.advanced.crowd_flow import CrowdFlowAnalyzer
from src.advanced.behavior import BehaviorPatternAnalyzer
from src.advanced.queue import QueueManagementSystem
from src.system.alerts import AlertSystem
from src.system.analytics import AnalyticsDashboard


@dataclass
class ProcessingConfig:
    """Configuration for a single processing run."""

    # Core features
    detection: bool = True
    segmentation: bool = False
    tracking: bool = True
    emotion: bool = False

    # Model parameters
    detection_conf: float = 0.5
    emotion_conf: float = 0.6

    # Advanced features
    enable_heatmap: bool = False
    enable_trajectories: bool = False
    enable_distancing: bool = False
    enable_flow: bool = False
    enable_behavior: bool = False
    enable_alerts: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProcessingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


@dataclass
class ProcessingResult:
    """Results from processing a single frame."""

    # Detection results
    detections: Any = None
    segmentation: Any = None
    tracks: Any = None
    track_info: List[Dict] = field(default_factory=list)

    # Face and emotion
    faces: List[Dict] = field(default_factory=list)
    emotions: List[Optional[Dict]] = field(default_factory=list)

    # Counts and metrics
    person_count: int = 0
    face_count: int = 0
    processing_time: float = 0.0
    confidences: List[float] = field(default_factory=list)

    # Advanced analysis
    distancing_violations: int = 0
    behavior_anomalies: int = 0

    # Alerts generated
    alerts: List[Dict] = field(default_factory=list)

    # Frame info
    frame_number: int = 0

    @property
    def boxes(self) -> List:
        """Get bounding boxes from detections."""
        if self.detections is not None and hasattr(self.detections, 'boxes'):
            return self.detections.boxes
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'person_count': self.person_count,
            'face_count': self.face_count,
            'processing_time': self.processing_time,
            'confidences': self.confidences,
            'emotions': [e for e in self.emotions if e is not None],
            'distancing_violations': self.distancing_violations,
            'behavior_anomalies': self.behavior_anomalies,
            'frame_number': self.frame_number,
            'track_count': len(self.track_info)
        }


class PipelineOrchestrator:
    """
    Orchestrates the full processing pipeline.

    Coordinates model inference, analysis, and data collection
    for each frame or image processed.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        model_loader: Optional[ModelLoader] = None,
        heatmap_gen: Optional[HeatmapGenerator] = None,
        trajectory_analyzer: Optional[TrajectoryAnalyzer] = None,
        roi_manager: Optional[ROIManager] = None,
        distancing_monitor: Optional[SocialDistancingMonitor] = None,
        crowd_flow_analyzer: Optional[CrowdFlowAnalyzer] = None,
        behavior_analyzer: Optional[BehaviorPatternAnalyzer] = None,
        queue_system: Optional[QueueManagementSystem] = None,
        alert_system: Optional[AlertSystem] = None,
        analytics: Optional[AnalyticsDashboard] = None
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Application configuration
            model_loader: ModelLoader instance for accessing models (created if not provided)
            heatmap_gen: HeatmapGenerator for density visualization
            trajectory_analyzer: TrajectoryAnalyzer for path tracking
            roi_manager: ROIManager for zone management
            distancing_monitor: SocialDistancingMonitor
            crowd_flow_analyzer: CrowdFlowAnalyzer
            behavior_analyzer: BehaviorPatternAnalyzer
            queue_system: QueueManagementSystem
            alert_system: AlertSystem for generating alerts
            analytics: AnalyticsDashboard for data collection
        """
        self.config = config or AppConfig.default()
        # Lazy model loader - created on demand
        self._model_loader = model_loader

        # Analyzers (also lazy - can be added later)
        self.heatmap = heatmap_gen
        self.trajectory = trajectory_analyzer
        self.roi = roi_manager
        self.distancing = distancing_monitor
        self.crowd_flow = crowd_flow_analyzer
        self.behavior = behavior_analyzer
        self.queue = queue_system

        # System components
        self.alerts = alert_system
        self.analytics = analytics

        # Analyzers dict for external access
        self.analyzers: Dict[str, Any] = {}

        # Frame counter
        self._frame_count = 0

    @property
    def model_loader(self) -> ModelLoader:
        """Lazy-load the model loader on first access."""
        if self._model_loader is None:
            self._model_loader = ModelLoader(self.config)
        return self._model_loader

    def process_frame(
        self,
        image: np.ndarray,
        config: Optional[ProcessingConfig] = None,
        frame_num: Optional[int] = None
    ) -> ProcessingResult:
        """
        Process a single frame through the pipeline.

        Args:
            image: Input image (BGR format)
            config: Processing configuration
            frame_num: Optional frame number override

        Returns:
            ProcessingResult with all analysis data
        """
        start_time = time.time()
        config = config or ProcessingConfig()

        if frame_num is None:
            frame_num = self._frame_count
            self._frame_count += 1

        result = ProcessingResult(frame_number=frame_num)

        # ===== Core Detection =====
        if config.detection:
            detector = self.model_loader.get_detector()
            result.detections = detector.predict(image, conf=config.detection_conf)

            if result.detections.boxes is not None:
                result.person_count = len(result.detections.boxes)
                result.confidences = [
                    float(box.conf[0]) for box in result.detections.boxes
                ]

        # ===== Segmentation =====
        if config.segmentation:
            segmenter = self.model_loader.get_segmenter()
            result.segmentation = segmenter.predict(image, conf=config.detection_conf)

        # ===== Tracking =====
        if config.tracking and result.detections is not None:
            tracker = self.model_loader.get_tracker()
            result.tracks = tracker.predict(result.detections, frame=image)

            if result.tracks:
                result.track_info = tracker.get_confirmed_tracks(result.tracks)

                # Update trajectory analyzer
                if self.trajectory and config.enable_trajectories:
                    self.trajectory.update_from_tracks(result.tracks)

                # Update crowd flow analyzer
                if self.crowd_flow and config.enable_flow:
                    self.crowd_flow.analyze_flow(result.tracks, image.shape)

        # ===== Face Detection and Emotion =====
        if config.emotion and result.detections is not None:
            face_detector = self.model_loader.get_face_detector()
            emotion_recognizer = self.model_loader.get_emotion_recognizer()

            # Detect faces within person bounding boxes
            if result.detections.boxes is not None:
                for box in result.detections.boxes:
                    person_bbox = box.xyxy[0].cpu().numpy()

                    # Find faces in this person region
                    faces = face_detector.detect_faces_in_person_bbox(image, person_bbox)

                    if faces:
                        # Use the first/largest face
                        face = faces[0]
                        result.faces.append(face)

                        # Recognize emotion
                        emotion = emotion_recognizer.predict(image, face['bbox'])
                        result.emotions.append(emotion)
                    else:
                        # Fallback: try emotion on person bbox
                        emotion = emotion_recognizer.predict(image, person_bbox)
                        result.emotions.append(emotion)

            result.face_count = len(result.faces)

        # ===== Heatmap Update =====
        if self.heatmap and config.enable_heatmap and result.detections is not None:
            self.heatmap.update_from_detections(result.detections)

        # ===== Social Distancing =====
        if self.distancing and config.enable_distancing and result.detections is not None:
            self.distancing.update(result.detections)
            result.distancing_violations = len(self.distancing.violations)

        # ===== Behavior Analysis =====
        if self.behavior and config.enable_behavior and self.trajectory:
            behaviors = self.behavior.analyze_from_trajectory_analyzer(self.trajectory)
            result.behavior_anomalies = len(self.behavior.get_anomalies())

        # ===== ROI Updates =====
        if self.roi and result.tracks:
            self.roi.update_from_tracks(result.tracks)

        # ===== Queue Updates =====
        if self.queue and result.tracks:
            self.queue.update(result.tracks)

        # Calculate processing time
        result.processing_time = time.time() - start_time

        # ===== Analytics =====
        if self.analytics:
            self.analytics.add_frame_data(
                frame_num,
                result.person_count,
                result.emotions,
                result.confidences,
                result.processing_time,
                result.distancing_violations,
                result.behavior_anomalies
            )

        # ===== Alerts =====
        if self.alerts and config.enable_alerts:
            self.alerts.check_crowding(result.person_count)

            if config.enable_distancing:
                self.alerts.check_social_distancing(result.distancing_violations)

            if self.trajectory:
                self.alerts.check_loitering(self.trajectory)

        return result

    def process_image(
        self,
        image: np.ndarray,
        config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """
        Process a single image (alias for process_frame).

        Args:
            image: Input image
            config: Processing configuration

        Returns:
            ProcessingResult
        """
        return self.process_frame(image, config, frame_num=0)

    def reset_analyzers(self) -> None:
        """Reset all analyzer states."""
        if self.heatmap:
            self.heatmap.reset()
        if self.trajectory:
            self.trajectory.reset()
        if self.distancing:
            self.distancing.reset()
        if self.crowd_flow:
            self.crowd_flow.reset()
        if self.behavior:
            self.behavior.reset()
        if self.roi:
            self.roi.reset()
        if self.queue:
            self.queue.reset()

        self._frame_count = 0

    def reset_analytics(self) -> None:
        """Reset analytics data."""
        if self.analytics:
            self.analytics.reset()

    def reset_alerts(self) -> None:
        """Clear all alerts."""
        if self.alerts:
            self.alerts.clear_alerts()

    def reset_all(self) -> None:
        """Reset everything."""
        self.reset_analyzers()
        self.reset_analytics()
        self.reset_alerts()

    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all analyzers.

        Returns:
            Combined statistics dictionary
        """
        stats = {}

        if self.analytics:
            stats['analytics'] = self.analytics.get_summary_stats()

        if self.heatmap:
            stats['heatmap'] = self.heatmap.get_statistics()

        if self.trajectory:
            stats['trajectory'] = self.trajectory.get_statistics()

        if self.distancing:
            stats['social_distancing'] = self.distancing.get_statistics()

        if self.crowd_flow:
            stats['crowd_flow'] = self.crowd_flow.get_statistics()

        if self.behavior:
            stats['behavior'] = self.behavior.get_statistics()

        if self.roi:
            stats['roi'] = self.roi.get_statistics()

        if self.queue:
            stats['queues'] = self.queue.get_statistics()

        if self.alerts:
            stats['alert_counts'] = self.alerts.get_alert_counts()

        return stats
