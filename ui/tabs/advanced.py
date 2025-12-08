"""
Advanced analysis tab for social distancing, crowd flow, and behavior analysis.
"""

from typing import Dict, Any, Optional, List
import streamlit as st
import numpy as np
import cv2

from src.advanced.social_distancing import SocialDistancingMonitor
from src.advanced.crowd_flow import CrowdFlowAnalyzer
from src.advanced.behavior import BehaviorPatternAnalyzer
from src.advanced.queue import QueueManagementSystem
from ui.components.charts import render_direction_chart, render_violations_chart


def render_advanced_tab(
    social_distancing: Optional[SocialDistancingMonitor] = None,
    crowd_flow: Optional[CrowdFlowAnalyzer] = None,
    behavior_analyzer: Optional[BehaviorPatternAnalyzer] = None,
    queue_system: Optional[QueueManagementSystem] = None
) -> None:
    """
    Render the advanced analysis tab.

    Args:
        social_distancing: Social distancing monitor instance
        crowd_flow: Crowd flow analyzer instance
        behavior_analyzer: Behavior pattern analyzer instance
        queue_system: Queue management system instance
    """
    st.header("🔬 Advanced Analysis")

    # Feature selection tabs
    feature_tab = st.tabs([
        "👥 Social Distancing",
        "🌊 Crowd Flow",
        "🎭 Behavior Analysis",
        "📋 Queue Management"
    ])

    with feature_tab[0]:
        _render_social_distancing_section(social_distancing)

    with feature_tab[1]:
        _render_crowd_flow_section(crowd_flow)

    with feature_tab[2]:
        _render_behavior_section(behavior_analyzer)

    with feature_tab[3]:
        _render_queue_section(queue_system)


def _render_social_distancing_section(
    monitor: Optional[SocialDistancingMonitor] = None
) -> None:
    """Render social distancing analysis section."""
    st.subheader("👥 Social Distancing Monitor")

    st.markdown("""
    Monitor compliance with social distancing guidelines by analyzing
    the distance between detected persons.
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        min_distance = st.slider(
            "Minimum Safe Distance (pixels)",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="Minimum distance between persons to be considered safe"
        )

    with col2:
        violation_threshold = st.slider(
            "Violation Alert Threshold",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of violations to trigger an alert"
        )

    # Display current statistics
    if monitor is not None and hasattr(monitor, 'history') and monitor.history:
        st.markdown("---")
        st.subheader("📊 Violation Statistics")

        # Calculate statistics
        total_violations = sum(monitor.history)
        avg_violations = total_violations / len(monitor.history) if monitor.history else 0
        max_violations = max(monitor.history) if monitor.history else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Violations", total_violations)
        with col2:
            st.metric("Avg per Frame", f"{avg_violations:.1f}")
        with col3:
            st.metric("Max Violations", max_violations)
        with col4:
            compliance_rate = 1.0 - (avg_violations / 10) if avg_violations < 10 else 0
            st.metric("Compliance Rate", f"{compliance_rate*100:.0f}%")

        # Violations over time chart
        frame_nums = list(range(1, len(monitor.history) + 1))
        render_violations_chart(frame_nums, monitor.history)
    else:
        st.info("Process video or images with social distancing enabled to see statistics.")


def _render_crowd_flow_section(
    analyzer: Optional[CrowdFlowAnalyzer] = None
) -> None:
    """Render crowd flow analysis section."""
    st.subheader("🌊 Crowd Flow Analysis")

    st.markdown("""
    Analyze movement patterns and flow directions of people in the scene.
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        flow_sensitivity = st.slider(
            "Flow Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Sensitivity for detecting movement direction"
        )

    with col2:
        min_track_length = st.slider(
            "Min Track Length",
            min_value=3,
            max_value=20,
            value=5,
            help="Minimum track length for flow analysis"
        )

    # Display flow statistics
    if analyzer is not None:
        stats = analyzer.get_statistics()
        flow_data = stats.get('direction_distribution', {})

        if flow_data and sum(flow_data.values()) > 0:
            st.markdown("---")
            st.subheader("📊 Flow Direction Distribution")
            render_direction_chart(flow_data, title="Crowd Movement Directions")

            # Flow summary
            st.markdown("---")
            st.subheader("📈 Flow Summary")

            dominant_direction = max(flow_data, key=flow_data.get) if flow_data else "N/A"
            total_movements = sum(flow_data.values())

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dominant Direction", dominant_direction)
            with col2:
                st.metric("Total Movements", total_movements)
            with col3:
                avg_speed = analyzer.get_average_speed() if hasattr(analyzer, 'get_average_speed') else 0
                st.metric("Avg Speed", f"{avg_speed:.1f} px/frame")
        else:
            st.info("Process video with tracking enabled to see flow analysis.")
    else:
        st.info("Enable crowd flow analysis in sidebar to collect data.")


def _render_behavior_section(
    analyzer: Optional[BehaviorPatternAnalyzer] = None
) -> None:
    """Render behavior analysis section."""
    st.subheader("🎭 Behavior Pattern Analysis")

    st.markdown("""
    Detect unusual behavior patterns such as:
    - **Loitering**: Person staying in same area for extended time
    - **Running**: Fast movement detection
    - **Gathering**: Groups forming in specific areas
    - **Counter-flow**: Movement against dominant direction
    """)

    # Configuration
    st.markdown("### Configuration")

    col1, col2 = st.columns(2)

    with col1:
        loitering_threshold = st.slider(
            "Loitering Time (frames)",
            min_value=30,
            max_value=300,
            value=90,
            help="Frames in same area to be considered loitering"
        )

        running_threshold = st.slider(
            "Running Speed (px/frame)",
            min_value=10,
            max_value=100,
            value=30,
            help="Speed threshold to detect running"
        )

    with col2:
        gathering_threshold = st.slider(
            "Gathering Size",
            min_value=3,
            max_value=20,
            value=5,
            help="Number of people to be considered a gathering"
        )

        proximity_threshold = st.slider(
            "Gathering Proximity (px)",
            min_value=50,
            max_value=200,
            value=100,
            help="Distance to consider people as part of same group"
        )

    # Display behavior statistics
    if analyzer is not None and hasattr(analyzer, 'behavior_counts'):
        behavior_counts = analyzer.behavior_counts

        if behavior_counts and sum(behavior_counts.values()) > 0:
            st.markdown("---")
            st.subheader("📊 Detected Behaviors")

            cols = st.columns(4)
            behavior_icons = {
                'loitering': '🚶',
                'running': '🏃',
                'gathering': '👥',
                'counter_flow': '↔️'
            }

            for i, (behavior, count) in enumerate(behavior_counts.items()):
                with cols[i % 4]:
                    icon = behavior_icons.get(behavior, '📍')
                    label = behavior.replace('_', ' ').title()
                    st.metric(f"{icon} {label}", count)

            # Recent detections
            if hasattr(analyzer, 'recent_detections') and analyzer.recent_detections:
                st.markdown("---")
                st.subheader("🕐 Recent Behavior Detections")

                for detection in analyzer.recent_detections[-10:]:
                    st.markdown(f"- **{detection['type']}** at frame {detection['frame']}: {detection['description']}")
        else:
            st.info("No unusual behaviors detected yet.")
    else:
        st.info("Enable behavior analysis and process video to detect patterns.")


def _render_queue_section(
    queue_system: Optional[QueueManagementSystem] = None
) -> None:
    """Render queue management section."""
    st.subheader("📋 Queue Management System")

    st.markdown("""
    Monitor and analyze queue formations for service optimization.
    """)

    # Queue zone configuration
    st.markdown("### Queue Zone Configuration")

    col1, col2 = st.columns(2)

    with col1:
        queue_enabled = st.checkbox("Enable Queue Monitoring", value=True)
        max_queue_length = st.slider(
            "Max Acceptable Queue Length",
            min_value=3,
            max_value=30,
            value=10,
            help="Alert when queue exceeds this length"
        )

    with col2:
        wait_time_threshold = st.slider(
            "Wait Time Alert (seconds)",
            min_value=30,
            max_value=600,
            value=120,
            help="Alert when estimated wait exceeds this time"
        )

    # Display queue statistics
    if queue_system is not None and hasattr(queue_system, 'queues'):
        st.markdown("---")
        st.subheader("📊 Queue Statistics")

        if queue_system.queues:
            for queue_id, queue_data in queue_system.queues.items():
                with st.expander(f"Queue {queue_id}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Current Length", queue_data.get('length', 0))
                    with col2:
                        avg_wait = queue_data.get('avg_wait_time', 0)
                        st.metric("Avg Wait", f"{avg_wait:.0f}s")
                    with col3:
                        throughput = queue_data.get('throughput', 0)
                        st.metric("Throughput/min", f"{throughput:.1f}")
                    with col4:
                        efficiency = queue_data.get('efficiency', 0)
                        st.metric("Efficiency", f"{efficiency*100:.0f}%")

            # Queue alerts
            if hasattr(queue_system, 'alerts') and queue_system.alerts:
                st.markdown("---")
                st.subheader("⚠️ Queue Alerts")
                for alert in queue_system.alerts[-5:]:
                    severity_color = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(alert.get('severity', 'low'), '⚪')
                    st.markdown(f"{severity_color} {alert.get('message', 'Alert')}")
        else:
            st.info("No queues detected. Define queue zones to start monitoring.")
    else:
        st.info("Enable queue management and define queue zones to start monitoring.")

    # Queue zone definition tool
    st.markdown("---")
    st.subheader("🎯 Define Queue Zones")

    st.markdown("""
    To define a queue zone:
    1. Upload an image or use a video frame
    2. Draw a rectangle around the queue area
    3. Save the zone configuration
    """)

    # Placeholder for zone drawing
    if st.button("📐 Draw Queue Zone"):
        st.info("Zone drawing feature - Use the ROI Manager in the processing pipeline to define queue zones.")
