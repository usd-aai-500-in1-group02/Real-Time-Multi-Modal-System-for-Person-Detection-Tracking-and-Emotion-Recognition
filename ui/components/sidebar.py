"""
Sidebar configuration component.
"""

from typing import Dict, Any
import streamlit as st

from src.config.settings import AppConfig
from src.pipeline.orchestrator import ProcessingConfig


def render_sidebar(config: AppConfig = None) -> Dict[str, Any]:
    """
    Render the sidebar configuration panel.

    Args:
        config: Application configuration

    Returns:
        Dictionary with user-selected configuration
    """
    config = config or AppConfig.default()

    st.sidebar.header("⚙️ Configuration")

    # Analysis Tasks
    st.sidebar.subheader("📋 Analysis Tasks")
    user_config = {
        'detection': st.sidebar.checkbox(
            "Person Detection",
            value=config.ui.default_detection,
            help="Required for all other features"
        ),
        'segmentation': st.sidebar.checkbox(
            "Instance Segmentation",
            value=config.ui.default_segmentation
        ),
        'tracking': st.sidebar.checkbox(
            "Multi-Object Tracking",
            value=config.ui.default_tracking
        ),
        'counting': st.sidebar.checkbox(
            "Person Counting",
            value=config.ui.default_counting
        ),
        'emotion': st.sidebar.checkbox(
            "Emotion Recognition",
            value=config.ui.default_emotion,
            help="Requires Person Detection to be enabled"
        ),
    }

    # Auto-enable detection if emotion/tracking/segmentation is enabled
    if user_config['emotion'] or user_config['tracking'] or user_config['segmentation']:
        if not user_config['detection']:
            st.sidebar.warning("⚠️ Person Detection auto-enabled (required for selected features)")
            user_config['detection'] = True

    # Model Parameters
    st.sidebar.subheader("🎛️ Model Parameters")
    user_config['detection_conf'] = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=config.model.detection_confidence,
        step=0.05
    )
    user_config['emotion_conf'] = st.sidebar.slider(
        "Emotion Confidence",
        min_value=0.0,
        max_value=1.0,
        value=config.model.emotion_confidence,
        step=0.05
    )
    user_config['iou_threshold'] = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.model.iou_threshold,
        step=0.05
    )

    # Advanced Features
    st.sidebar.subheader("🔬 Advanced Features")
    user_config['show_heatmap'] = st.sidebar.checkbox(
        "🔥 Show Heatmap",
        value=config.ui.default_heatmap
    )
    user_config['show_trajectories'] = st.sidebar.checkbox(
        "📈 Show Trajectories",
        value=config.ui.default_trajectories
    )
    user_config['show_distancing'] = st.sidebar.checkbox(
        "👥 Social Distancing",
        value=config.ui.default_distancing
    )
    user_config['show_flow'] = st.sidebar.checkbox(
        "🌊 Crowd Flow",
        value=config.ui.default_flow
    )
    user_config['enable_alerts'] = st.sidebar.checkbox(
        "🚨 Enable Alerts",
        value=config.ui.default_alerts
    )

    # Alert Settings (if alerts enabled)
    if user_config['enable_alerts']:
        st.sidebar.subheader("🚨 Alert Settings")
        user_config['crowding_threshold'] = st.sidebar.slider(
            "Crowding Threshold",
            min_value=1,
            max_value=50,
            value=config.alerts.crowding_threshold
        )
        user_config['enable_distancing_alerts'] = st.sidebar.checkbox(
            "Enable Social Distancing Alerts",
            value=config.alerts.social_distancing_enabled
        )

        if user_config['enable_distancing_alerts']:
            user_config['min_distance'] = st.sidebar.slider(
                "Min Distance (pixels)",
                min_value=50,
                max_value=300,
                value=config.alerts.min_social_distance,
                step=10
            )

    return user_config


def get_processing_config(user_config: Dict[str, Any]) -> ProcessingConfig:
    """
    Convert user config to ProcessingConfig.

    Args:
        user_config: Dictionary from render_sidebar()

    Returns:
        ProcessingConfig object
    """
    return ProcessingConfig(
        detection=user_config.get('detection', True),
        segmentation=user_config.get('segmentation', False),
        tracking=user_config.get('tracking', True),
        emotion=user_config.get('emotion', False),
        detection_conf=user_config.get('detection_conf', 0.5),
        emotion_conf=user_config.get('emotion_conf', 0.6),
        enable_heatmap=user_config.get('show_heatmap', False),
        enable_trajectories=user_config.get('show_trajectories', False),
        enable_distancing=user_config.get('show_distancing', False),
        enable_flow=user_config.get('show_flow', False),
        enable_alerts=user_config.get('enable_alerts', True),
    )


@st.cache_data
def _get_device_info() -> str:
    """Get device info (cached to avoid repeated torch imports)."""
    try:
        import torch
        return "CUDA" if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"


def render_sidebar_info() -> None:
    """Render sidebar information section."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")

    device = _get_device_info()
    st.sidebar.info(f"Running on: **{device}**")

    st.sidebar.markdown("""
    **Quick Tips:**
    - Enable tracking for video analysis
    - Use heatmap for crowd density visualization
    - Enable alerts for automated monitoring
    """)
