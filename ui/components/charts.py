"""
Chart components using Plotly.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render_person_count_chart(
    frame_numbers: List[int],
    person_counts: List[int],
    title: str = "Person Count Over Time"
) -> None:
    """
    Render person count timeline chart.

    Args:
        frame_numbers: List of frame numbers
        person_counts: List of person counts
        title: Chart title
    """
    if not frame_numbers or not person_counts:
        st.info("No data available for chart")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=person_counts,
        mode='lines+markers',
        name='Person Count',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Frame Number',
        yaxis_title='Person Count',
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="person_count_chart")


def render_emotion_pie_chart(
    emotion_data: Dict[str, int],
    title: str = "Emotion Distribution"
) -> None:
    """
    Render emotion distribution pie chart.

    Args:
        emotion_data: Dictionary of emotion: count
        title: Chart title
    """
    if not emotion_data or sum(emotion_data.values()) == 0:
        st.info("No emotion data available")
        return

    colors = {
        'happy': '#00c853',
        'sad': '#2196f3',
        'angry': '#f44336',
        'neutral': '#9e9e9e',
        'surprise': '#ffeb3b',
        'fear': '#9c27b0',
        'disgust': '#795548'
    }

    color_sequence = [colors.get(e, '#667eea') for e in emotion_data.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=list(emotion_data.keys()),
        values=list(emotion_data.values()),
        hole=0.3,
        marker_colors=color_sequence
    )])

    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="emotion_pie_chart")


def render_performance_chart(
    frame_numbers: List[int],
    processing_times: List[float],
    title: str = "Processing Performance"
) -> None:
    """
    Render processing performance chart.

    Args:
        frame_numbers: List of frame numbers
        processing_times: List of processing times in seconds
        title: Chart title
    """
    if not frame_numbers or not processing_times:
        st.info("No performance data available")
        return

    # Convert to FPS
    fps_values = [1.0 / t if t > 0 else 0 for t in processing_times]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=fps_values,
        mode='lines',
        name='FPS',
        line=dict(color='#00c853', width=2)
    ))

    # Add threshold line
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Target FPS (30)"
    )

    fig.update_layout(
        title=title,
        xaxis_title='Frame Number',
        yaxis_title='FPS',
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="performance_chart")


def render_violations_chart(
    frame_numbers: List[int],
    violations: List[int],
    title: str = "Social Distancing Violations"
) -> None:
    """
    Render violations over time chart.

    Args:
        frame_numbers: List of frame numbers
        violations: List of violation counts
        title: Chart title
    """
    if not frame_numbers or not violations:
        st.info("No violation data available")
        return

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=frame_numbers,
        y=violations,
        name='Violations',
        marker_color='rgba(255, 0, 0, 0.6)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Frame Number',
        yaxis_title='Violations',
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="violations_chart")


def render_direction_chart(
    direction_data: Dict[str, int],
    title: str = "Crowd Flow Direction"
) -> None:
    """
    Render crowd flow direction radar chart.

    Args:
        direction_data: Dictionary of direction: count
        title: Chart title
    """
    if not direction_data or sum(direction_data.values()) == 0:
        st.info("No direction data available")
        return

    directions = ['North', 'North-East', 'East', 'South-East',
                  'South', 'South-West', 'West', 'North-West']

    values = [direction_data.get(d, 0) for d in directions]
    values.append(values[0])  # Close the polygon

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=directions + [directions[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line_color='#667eea'
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="direction_chart")


def render_heatmap_chart(
    data: List[List[float]],
    title: str = "Density Heatmap"
) -> None:
    """
    Render density heatmap.

    Args:
        data: 2D list of density values
        title: Chart title
    """
    if not data:
        st.info("No heatmap data available")
        return

    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='Hot',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")


def render_multi_metric_chart(
    frame_numbers: List[int],
    metrics: Dict[str, List[float]],
    title: str = "Multi-Metric Analysis"
) -> None:
    """
    Render multiple metrics on the same chart.

    Args:
        frame_numbers: List of frame numbers
        metrics: Dictionary of metric_name: values list
        title: Chart title
    """
    if not frame_numbers or not metrics:
        st.info("No data available")
        return

    fig = go.Figure()

    colors = ['#667eea', '#00c853', '#f44336', '#ffeb3b', '#9c27b0']

    for i, (name, values) in enumerate(metrics.items()):
        fig.add_trace(go.Scatter(
            x=frame_numbers[:len(values)],
            y=values,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Frame Number',
        yaxis_title='Value',
        template='plotly_dark',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key="multi_metric_chart")
