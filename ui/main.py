"""
Main Streamlit application orchestration.
"""

import streamlit as st
from typing import Dict, Any, Optional

from src.config.settings import AppConfig
from src.models.loader import ModelLoader
from src.pipeline.orchestrator import PipelineOrchestrator
from src.system.analytics import AnalyticsDashboard
from src.system.alerts import AlertSystem

from ui.styles import apply_custom_styles
from ui.components.sidebar import render_sidebar, get_processing_config, render_sidebar_info
from ui.tabs.upload import render_upload_tab
from ui.tabs.webcam import render_webcam_tab
from ui.tabs.analytics import render_analytics_tab
from ui.tabs.reports import render_reports_tab


def main():
    """Main entry point for the Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Multi-Modal Person Detection & Analysis",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styling
    apply_custom_styles()

    # Initialize session state
    _initialize_session_state()

    # Load configuration
    config = AppConfig.default()

    # Render sidebar and get user configuration
    user_config = render_sidebar(config)
    render_sidebar_info()

    # Get processing configuration
    processing_config = get_processing_config(user_config)

    # Get or create orchestrator (models load lazily on first use)
    orchestrator = _get_orchestrator()

    # Main title
    st.title("👁️ Multi-Modal Person Detection & Analysis System")
    st.markdown("*Real-time detection, tracking, segmentation, and emotion analysis*")

    # Tab navigation
    tabs = st.tabs([
        "📤 Upload",
        "📹 Webcam",
        "📊 Analytics",
        "📑 Reports"
    ])

    # Tab 1: Upload
    with tabs[0]:
        render_upload_tab(orchestrator, processing_config, user_config)

    # Tab 2: Webcam
    with tabs[1]:
        render_webcam_tab(orchestrator, processing_config, user_config)

    # Tab 3: Analytics
    with tabs[2]:
        analytics_dashboard = _get_analytics_dashboard()
        render_analytics_tab(analytics_dashboard)

    # Tab 4: Reports
    with tabs[3]:
        analytics_dashboard = _get_analytics_dashboard()
        render_reports_tab(analytics_dashboard)

    # Footer
    _render_footer()


def _initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    # Model state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

    # Analytics state
    if 'analytics_dashboard' not in st.session_state:
        st.session_state.analytics_dashboard = AnalyticsDashboard()

    # Alert system
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = AlertSystem()

    # Processing history
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []

    # Advanced analyzers
    if 'social_distancing_monitor' not in st.session_state:
        st.session_state.social_distancing_monitor = None

    if 'crowd_flow_analyzer' not in st.session_state:
        st.session_state.crowd_flow_analyzer = None

    if 'behavior_analyzer' not in st.session_state:
        st.session_state.behavior_analyzer = None

    if 'queue_system' not in st.session_state:
        st.session_state.queue_system = None

    # Report history
    if 'report_history' not in st.session_state:
        st.session_state.report_history = []


@st.cache_resource(show_spinner=False)
def _get_orchestrator(_config_hash: str = "") -> PipelineOrchestrator:
    """
    Get or create the pipeline orchestrator.

    Uses Streamlit's cache_resource for model persistence.
    Models are loaded lazily when first needed, not at startup.
    """
    config = AppConfig.default()
    # Create orchestrator without loading models yet
    # Models will be lazy-loaded on first process_frame() call
    return PipelineOrchestrator(config=config)


def _get_analytics_dashboard() -> AnalyticsDashboard:
    """Get the analytics dashboard from session state."""
    return st.session_state.analytics_dashboard


def _render_footer() -> None:
    """Render application footer."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Technologies:**")
        st.markdown("YOLOv8 | DeepSort | DeepFace | MTCNN")

    with col2:
        st.markdown("**Features:**")
        st.markdown("Detection | Tracking | Segmentation | Emotion")

    with col3:
        st.markdown("**Built with:**")
        st.markdown("Streamlit | OpenCV | PyTorch")


if __name__ == "__main__":
    main()
