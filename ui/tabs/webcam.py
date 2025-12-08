"""
Webcam tab for live video processing with WebRTC.
"""

from typing import Dict, Any, Optional
import streamlit as st
import numpy as np
import cv2
import time

from src.pipeline.orchestrator import PipelineOrchestrator, ProcessingConfig, ProcessingResult
from src.visualization.drawer import ResultDrawer
from ui.components.metrics import render_metric_cards
from ui.components.alerts_display import render_alerts


def render_webcam_tab(
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """
    Render the webcam tab for live video processing.

    Args:
        orchestrator: Pipeline orchestrator instance
        processing_config: Processing configuration
        user_config: User configuration from sidebar
    """
    st.header("📹 Live Webcam Analysis")

    # Check for WebRTC availability
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        _render_webrtc_webcam(orchestrator, processing_config, user_config)
    except ImportError:
        st.warning("WebRTC not available. Using fallback OpenCV webcam.")
        _render_opencv_webcam(orchestrator, processing_config, user_config)


def _render_webrtc_webcam(
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """Render webcam using WebRTC (preferred for web deployment)."""
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    from ui.processors.webcam_processor import WebcamVideoProcessor

    # WebRTC configuration
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Create processor factory
    def processor_factory():
        return WebcamVideoProcessor(orchestrator, processing_config)

    # Webcam streamer
    ctx = webrtc_streamer(
        key="webcam-processor",
        video_processor_factory=processor_factory,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        async_processing=True
    )

    # Display live metrics
    if ctx.state.playing:
        st.markdown("---")
        _render_live_metrics_placeholder()

    # Instructions
    st.markdown("""
    **Instructions:**
    1. Click **START** to begin webcam capture
    2. Allow camera access when prompted
    3. View real-time analysis results
    4. Click **STOP** to end capture
    """)


def _render_opencv_webcam(
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """Render webcam using OpenCV (fallback for local development)."""
    # Session state for webcam
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    col1, col2 = st.columns(2)

    with col1:
        start_btn = st.button("▶️ Start Webcam", disabled=st.session_state.webcam_running)
    with col2:
        stop_btn = st.button("⏹️ Stop Webcam", disabled=not st.session_state.webcam_running)

    if start_btn:
        st.session_state.webcam_running = True
        st.rerun()

    if stop_btn:
        st.session_state.webcam_running = False
        st.rerun()

    if st.session_state.webcam_running:
        _run_opencv_capture(orchestrator, processing_config, user_config)


def _run_opencv_capture(
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """Run OpenCV webcam capture loop."""
    # Placeholders
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    alerts_placeholder = st.empty()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam. Please check camera permissions.")
        st.session_state.webcam_running = False
        return

    drawer = ResultDrawer()

    try:
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        while st.session_state.webcam_running:
            ret, frame = cap.read()

            if not ret:
                st.warning("Failed to read frame from webcam")
                break

            frame_count += 1

            # Process frame
            result = orchestrator.process_frame(frame, processing_config)

            # Draw results
            output_frame = drawer.draw_full_results(
                frame.copy(),
                result,
                show_emotions=processing_config.emotion,
                show_masks=processing_config.segmentation
            )

            # Calculate FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Draw FPS on frame
            cv2.putText(
                output_frame,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display frame
            output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(output_rgb, channels="RGB", use_container_width=True)

            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Persons", result.person_count)
                with col2:
                    st.metric("FPS", f"{current_fps:.1f}")
                with col3:
                    st.metric("Frame", frame_count)
                with col4:
                    st.metric("Proc Time", f"{result.processing_time*1000:.0f}ms")

            # Display alerts
            if user_config.get('enable_alerts') and result.alerts:
                with alerts_placeholder.container():
                    render_alerts(result.alerts[-5:], title="Live Alerts")

            # Small delay to prevent UI blocking
            time.sleep(0.01)

    finally:
        cap.release()


def _render_live_metrics_placeholder() -> None:
    """Render placeholder for live metrics display."""
    st.subheader("📊 Live Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Persons Detected", "—")
    with col2:
        st.metric("Processing FPS", "—")
    with col3:
        st.metric("Avg Confidence", "—")
    with col4:
        st.metric("Active Tracks", "—")

    st.info("Metrics will update in real-time when processing begins.")
