"""
Metric display components.
"""

from typing import Dict, Any, List, Optional
import streamlit as st

from src.pipeline.orchestrator import ProcessingResult


def render_metric_cards(result: ProcessingResult) -> None:
    """
    Render metric cards for processing results.

    Args:
        result: ProcessingResult object
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Persons", result.person_count)

    with col2:
        st.metric("Processing Time", f"{result.processing_time:.2f}s")

    with col3:
        emotions_found = len([e for e in result.emotions if e is not None])
        st.metric("Emotions Detected", emotions_found)

    with col4:
        avg_conf = sum(result.confidences) / len(result.confidences) if result.confidences else 0
        st.metric("Avg Confidence", f"{avg_conf:.2f}")


def render_metric_row(
    metrics: Dict[str, Any],
    columns: int = 4
) -> None:
    """
    Render a row of metrics.

    Args:
        metrics: Dictionary of metric_name: value pairs
        columns: Number of columns
    """
    cols = st.columns(columns)
    items = list(metrics.items())

    for i, (name, value) in enumerate(items):
        with cols[i % columns]:
            if isinstance(value, float):
                st.metric(name, f"{value:.2f}")
            else:
                st.metric(name, value)


def render_summary_stats(stats: Dict[str, Any]) -> None:
    """
    Render summary statistics.

    Args:
        stats: Statistics dictionary from AnalyticsDashboard
    """
    if not stats:
        st.info("No statistics available yet")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Frames", stats.get('total_frames', 0))
    with col2:
        st.metric("Avg Persons", f"{stats.get('avg_persons', 0):.2f}")
    with col3:
        st.metric("Max Persons", stats.get('max_persons', 0))
    with col4:
        st.metric("Avg FPS", f"{stats.get('avg_fps', 0):.1f}")


def render_detection_metrics(metrics: Dict[str, Any]) -> None:
    """
    Render detection performance metrics.

    Args:
        metrics: Detection metrics dictionary
    """
    st.subheader("📊 Detection Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Precision",
            f"{metrics.get('precision', 0):.2%}",
            help="True positives / (True positives + False positives)"
        )
    with col2:
        st.metric(
            "Recall",
            f"{metrics.get('recall', 0):.2%}",
            help="True positives / (True positives + False negatives)"
        )
    with col3:
        st.metric(
            "F1 Score",
            f"{metrics.get('f1_score', 0):.2%}",
            help="Harmonic mean of precision and recall"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "mAP@50",
            f"{metrics.get('mAP@50', 0):.2%}",
            help="Mean Average Precision at IoU=0.50"
        )
    with col5:
        st.metric(
            "mAP@75",
            f"{metrics.get('mAP@75', 0):.2%}",
            help="Mean Average Precision at IoU=0.75"
        )
    with col6:
        st.metric(
            "mAP@50:95",
            f"{metrics.get('mAP@50:95', 0):.2%}",
            help="Mean Average Precision across IoU thresholds"
        )


def render_tracking_metrics(metrics: Dict[str, Any]) -> None:
    """
    Render tracking performance metrics.

    Args:
        metrics: Tracking metrics dictionary
    """
    st.subheader("🎯 Tracking Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MOTA",
            f"{metrics.get('MOTA', 0):.2%}",
            help="Multiple Object Tracking Accuracy"
        )
    with col2:
        st.metric(
            "MOTP",
            f"{metrics.get('MOTP', 0):.2%}",
            help="Multiple Object Tracking Precision"
        )
    with col3:
        st.metric(
            "IDF1",
            f"{metrics.get('IDF1', 0):.2%}",
            help="ID F1 Score"
        )
    with col4:
        st.metric(
            "ID Switches",
            metrics.get('ID_Switches', 0),
            help="Number of identity switches"
        )


def render_emotion_table(emotions: List[Optional[Dict]]) -> None:
    """
    Render emotion analysis results as a table.

    Args:
        emotions: List of emotion dictionaries
    """
    import pandas as pd

    if not emotions or all(e is None for e in emotions):
        st.info("No emotions detected")
        return

    data = []
    for i, emotion in enumerate(emotions):
        if emotion is not None:
            data.append({
                'Person ID': i + 1,
                'Emotion': emotion.get('emotion', 'Unknown'),
                'Confidence': f"{emotion.get('confidence', 0) * 100:.1f}%"
            })

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
