"""
Centralized configuration settings for the Multi-Modal Person Analysis System.
All configuration dataclasses are defined here to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelSettings:
    """Configuration for ML models."""

    # YOLO Detection
    yolo_detect_model: str = "yolov8m.pt"
    detection_confidence: float = 0.5
    iou_threshold: float = 0.45
    person_class_id: int = 0

    # YOLO Segmentation
    yolo_segment_model: str = "yolov8m-seg.pt"

    # DeepFace Emotion
    emotion_confidence: float = 0.6
    emotion_detector_backend: str = "mtcnn"  # Using MTCNN for better face detection
    emotion_actions: tuple = ("emotion",)

    # Face Detection
    face_detector_backend: str = "mtcnn"


@dataclass
class TrackerSettings:
    """Configuration for DeepSort tracker."""

    max_age: int = 30
    n_init: int = 3
    nms_max_overlap: float = 1.0
    max_cosine_distance: float = 0.3
    nn_budget: Optional[int] = None


@dataclass
class AlertSettings:
    """Configuration for alert system thresholds."""

    # Crowding
    crowding_threshold: int = 10
    crowding_enabled: bool = True

    # Loitering
    loitering_duration: int = 30  # seconds
    loitering_enabled: bool = True

    # Social Distancing
    min_social_distance: int = 150  # pixels
    social_distancing_enabled: bool = False

    # Unusual Behavior
    unusual_behavior_enabled: bool = False


@dataclass
class AnalyzerSettings:
    """Configuration for analyzers."""

    # Heatmap
    heatmap_sigma: float = 50.0
    heatmap_alpha: float = 0.6

    # Trajectory
    trajectory_max_history: int = 50
    loitering_threshold_distance: float = 50.0
    loitering_min_frames: int = 30

    # Social Distancing
    social_distance_min: int = 150

    # Crowd Flow
    crowd_flow_grid_size: tuple = (4, 4)

    # Behavior Analysis
    stationary_speed_threshold: float = 0.1
    erratic_variance_threshold: float = 5.0


@dataclass
class VideoSettings:
    """Configuration for video processing."""

    target_fps: int = 30
    output_codec: str = "mp4v"
    skip_frames: int = 0  # Process every Nth frame (0 = no skip)


@dataclass
class UISettings:
    """Configuration for UI defaults."""

    default_detection: bool = True
    default_segmentation: bool = False
    default_tracking: bool = True
    default_counting: bool = True
    default_emotion: bool = False
    default_heatmap: bool = False
    default_trajectories: bool = False
    default_distancing: bool = False
    default_flow: bool = False
    default_alerts: bool = True


@dataclass
class AppConfig:
    """Main application configuration combining all settings."""

    model: ModelSettings = field(default_factory=ModelSettings)
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    analyzers: AnalyzerSettings = field(default_factory=AnalyzerSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    ui: UISettings = field(default_factory=UISettings)

    @classmethod
    def default(cls) -> "AppConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelSettings(**config_dict.get("model", {})),
            tracker=TrackerSettings(**config_dict.get("tracker", {})),
            alerts=AlertSettings(**config_dict.get("alerts", {})),
            analyzers=AnalyzerSettings(**config_dict.get("analyzers", {})),
            video=VideoSettings(**config_dict.get("video", {})),
            ui=UISettings(**config_dict.get("ui", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
