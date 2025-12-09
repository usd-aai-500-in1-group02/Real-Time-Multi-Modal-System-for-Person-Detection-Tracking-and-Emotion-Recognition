"""
Upload tab for image and video file processing.
"""

from typing import Dict, Any, Optional
import tempfile
import os
import streamlit as st
import numpy as np
from PIL import Image
import cv2

from src.pipeline.orchestrator import PipelineOrchestrator, ProcessingConfig, ProcessingResult
from src.pipeline.video_processor import VideoFileProcessor
from src.visualization.drawer import ResultDrawer
from src.visualization.overlays import OverlayManager
from ui.components.metrics import render_metric_cards, render_emotion_table
from ui.components.charts import (
    render_person_count_chart,
    render_emotion_pie_chart,
    render_performance_chart,
)


def render_upload_tab(
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """
    Render the upload tab for image and video processing.

    Args:
        orchestrator: Pipeline orchestrator instance
        processing_config: Processing configuration
        user_config: User configuration from sidebar
    """
    st.header("📤 Upload & Analyze")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: JPG, PNG for images; MP4, AVI, MOV, MKV for videos"
    )

    if uploaded_file is not None:
        file_type = _get_file_type(uploaded_file.name)

        if file_type == 'image':
            _process_image(uploaded_file, orchestrator, processing_config, user_config)
        elif file_type == 'video':
            _process_video(uploaded_file, orchestrator, processing_config, user_config)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")


def _get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = filename.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
        return 'image'
    elif ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
        return 'video'
    return 'unknown'


def _process_image(
    uploaded_file,
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """Process uploaded image file."""
    # Load image
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_array

    # Display columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    # Process image
    with st.spinner("Processing image..."):
        result = orchestrator.process_frame(image_bgr, processing_config)

    # Add to analytics dashboard
    if 'analytics_dashboard' in st.session_state:
        # Extract confidences from detections
        confidences = []
        if result.detections is not None and hasattr(result.detections, 'boxes'):
            confidences = [float(box.conf[0]) for box in result.detections.boxes]

        st.session_state.analytics_dashboard.add_frame_data(
            frame_num=1,  # Single image
            person_count=result.person_count,
            emotions=result.emotions,
            confidences=confidences,
            proc_time=result.processing_time,
            violations=0,
            anomalies=0
        )

    # Draw results
    drawer = ResultDrawer()
    output_image = drawer.draw_full_results(
        image_bgr.copy(),
        result,
        show_emotions=processing_config.emotion,
        show_masks=processing_config.segmentation,
        show_distancing=user_config.get('show_distancing', False),
        min_distance=user_config.get('min_distance', 150.0)
    )

    # Apply overlays if enabled
    if user_config.get('show_heatmap') and result.detections is not None:
        # Create heatmap analyzer if not exists
        if orchestrator.heatmap is None:
            from src.analyzers.heatmap import HeatmapGenerator
            orchestrator.heatmap = HeatmapGenerator(frame_shape=image_bgr.shape)

        # Update heatmap with detections
        orchestrator.heatmap.update_from_detections(result.detections)

        # Apply heatmap overlay
        output_image = orchestrator.heatmap.draw(output_image, alpha=0.5)

    with col2:
        st.subheader("🔍 Analysis Results")
        # Convert BGR to RGB for display
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        st.image(output_rgb, use_container_width=True)

    # Display metrics
    st.markdown("---")
    render_metric_cards(result)

    # Emotion analysis section
    if processing_config.emotion and result.emotions:
        st.markdown("---")
        st.subheader("😊 Emotion Analysis")

        col1, col2 = st.columns(2)

        with col1:
            render_emotion_table(result.emotions)

        with col2:
            # Count emotions for pie chart
            emotion_counts = {}
            for emotion_data in result.emotions:
                if emotion_data is not None:
                    emotion = emotion_data.get('emotion', 'unknown')
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if emotion_counts:
                render_emotion_pie_chart(emotion_counts)

    # Download button for processed image
    st.markdown("---")
    _render_download_button(output_image, "processed_image.jpg")


def _process_video(
    uploaded_file,
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any]
) -> None:
    """Process uploaded video file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    try:
        # Video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Display video info
        st.subheader("📹 Video Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("Frames", total_frames)
        with col3:
            st.metric("FPS", f"{fps:.1f}")
        with col4:
            st.metric("Resolution", f"{width}x{height}")

        # Processing options
        st.subheader("⚙️ Processing Options")
        col1, col2 = st.columns(2)

        with col1:
            process_every_n = st.slider(
                "Process every N frames",
                min_value=1,
                max_value=10,
                value=1,
                help="Skip frames for faster processing"
            )

        with col2:
            max_frames = st.slider(
                "Max frames to process",
                min_value=10,
                max_value=min(total_frames, 500),
                value=min(total_frames, 100),
                help="Limit frames for faster processing"
            )

        # Process button
        if st.button("🚀 Process Video", type="primary"):
            _run_video_processing(
                video_path,
                orchestrator,
                processing_config,
                user_config,
                process_every_n,
                max_frames
            )

    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            os.unlink(video_path)


def _run_video_processing(
    video_path: str,
    orchestrator: PipelineOrchestrator,
    processing_config: ProcessingConfig,
    user_config: Dict[str, Any],
    process_every_n: int,
    max_frames: int
) -> None:
    """Run video processing with progress tracking."""
    # Initialize video processor
    video_processor = VideoFileProcessor(orchestrator)

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_display = st.empty()

    # Results storage
    results_data = {
        'frame_numbers': [],
        'person_counts': [],
        'processing_times': [],
        'emotion_counts': {},
        'frames_processed': 0
    }

    def progress_callback(frame_num: int, total: int, result: ProcessingResult, frame: np.ndarray):
        """Callback for progress updates."""
        progress = frame_num / total
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_num}/{total}")

        # Store results
        results_data['frame_numbers'].append(frame_num)
        results_data['person_counts'].append(result.person_count)
        results_data['processing_times'].append(result.processing_time)
        results_data['frames_processed'] += 1

        # Count emotions
        for emotion_data in result.emotions:
            if emotion_data is not None:
                emotion = emotion_data.get('emotion', 'unknown')
                results_data['emotion_counts'][emotion] = \
                    results_data['emotion_counts'].get(emotion, 0) + 1

        # Add to analytics dashboard
        if 'analytics_dashboard' in st.session_state:
            # Extract confidences from detections
            confidences = []
            if result.detections is not None and hasattr(result.detections, 'boxes'):
                confidences = [float(box.conf[0]) for box in result.detections.boxes]

            st.session_state.analytics_dashboard.add_frame_data(
                frame_num=frame_num,
                person_count=result.person_count,
                emotions=result.emotions,
                confidences=confidences,
                proc_time=result.processing_time,
                violations=0,  # TODO: Add violations if available
                anomalies=0  # TODO: Add anomalies if available
            )

        # Draw and display frame
        drawer = ResultDrawer()
        output_frame = drawer.draw_full_results(
            frame.copy(),
            result,
            show_emotions=processing_config.emotion,
            show_masks=processing_config.segmentation,
            show_distancing=user_config.get('show_distancing', False),
            min_distance=user_config.get('min_distance', 150.0)
        )

        # Apply overlays if enabled
        if user_config.get('show_heatmap') and result.detections is not None:
            # Create heatmap analyzer if not exists
            if orchestrator.heatmap is None:
                from src.analyzers.heatmap import HeatmapGenerator
                orchestrator.heatmap = HeatmapGenerator(frame_shape=frame.shape)

            # Update heatmap with detections
            orchestrator.heatmap.update_from_detections(result.detections)

            # Apply heatmap overlay
            output_frame = orchestrator.heatmap.draw(output_frame, alpha=0.5)

        # Apply trajectories if enabled
        if user_config.get('show_trajectories') and result.tracks:
            if orchestrator.trajectory is None:
                from src.analyzers.trajectory import TrajectoryAnalyzer
                orchestrator.trajectory = TrajectoryAnalyzer()

            # Update trajectories
            orchestrator.trajectory.update_from_tracks(result.tracks)

            # Draw trajectories
            output_frame = orchestrator.trajectory.draw(output_frame)

        output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        frame_display.image(output_rgb, use_container_width=True, caption=f"Frame {frame_num}")

    # Process video
    with st.spinner("Processing video..."):
        video_processor.process_video(
            video_path,
            processing_config,
            progress_callback=progress_callback,
            skip_frames=process_every_n - 1,
            max_frames=max_frames
        )

    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    # Display results
    st.markdown("---")
    st.subheader("📊 Video Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames Processed", results_data['frames_processed'])
    with col2:
        avg_persons = sum(results_data['person_counts']) / len(results_data['person_counts']) \
            if results_data['person_counts'] else 0
        st.metric("Avg Persons", f"{avg_persons:.1f}")
    with col3:
        max_persons = max(results_data['person_counts']) if results_data['person_counts'] else 0
        st.metric("Max Persons", max_persons)
    with col4:
        avg_time = sum(results_data['processing_times']) / len(results_data['processing_times']) \
            if results_data['processing_times'] else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        st.metric("Avg FPS", f"{avg_fps:.1f}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        render_person_count_chart(
            results_data['frame_numbers'],
            results_data['person_counts'],
            title="Person Count Over Time",
            key="upload_person_count"
        )

    with col2:
        render_performance_chart(
            results_data['frame_numbers'],
            results_data['processing_times'],
            title="Processing Performance",
            key="upload_performance"
        )

    # Emotion distribution
    if results_data['emotion_counts']:
        st.markdown("---")
        render_emotion_pie_chart(
            results_data['emotion_counts'],
            title="Emotion Distribution Across Video"
        )


def _render_download_button(image: np.ndarray, filename: str) -> None:
    """Render download button for processed image."""
    # Encode image
    success, encoded = cv2.imencode('.jpg', image)
    if success:
        st.download_button(
            label="📥 Download Processed Image",
            data=encoded.tobytes(),
            file_name=filename,
            mime="image/jpeg"
        )
