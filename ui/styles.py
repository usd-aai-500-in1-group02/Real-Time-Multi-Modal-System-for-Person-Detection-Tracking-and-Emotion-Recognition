"""
CSS styles for the Streamlit UI.
Extracted from app4.py lines 413-451.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
.main {
    background: linear-gradient(135deg, #1e1e2e 0%, #2d1b4e 100%);
}

.stButton>button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    transition: transform 0.2s, box-shadow 0.2s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

.alert-box {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: 600;
}

.alert-high {
    background: rgba(255, 0, 0, 0.2);
    border-left: 4px solid #ff0000;
    color: #ff6b6b;
}

.alert-medium {
    background: rgba(255, 165, 0, 0.2);
    border-left: 4px solid #ffa500;
    color: #ffc107;
}

.alert-low {
    background: rgba(255, 255, 0, 0.2);
    border-left: 4px solid #ffff00;
    color: #ffeb3b;
}

.stat-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(102, 126, 234, 0.3);
    text-align: center;
}

.header-gradient {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.success-badge {
    background: linear-gradient(90deg, #00c853, #64dd17);
    padding: 5px 15px;
    border-radius: 20px;
    color: white;
    font-size: 0.85em;
}

.warning-badge {
    background: linear-gradient(90deg, #ff9800, #ffc107);
    padding: 5px 15px;
    border-radius: 20px;
    color: black;
    font-size: 0.85em;
}

.danger-badge {
    background: linear-gradient(90deg, #f44336, #e91e63);
    padding: 5px 15px;
    border-radius: 20px;
    color: white;
    font-size: 0.85em;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #667eea, #764ba2);
    border-radius: 4px;
}
</style>
"""


def apply_custom_styles() -> None:
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(
    title: str = "Multi-Modal Person Analysis System",
    subtitle: str = "Detection • Segmentation • Tracking • Emotion • Social Distancing • Crowd Flow • Behavior Analysis",
    icon: str = "👁️"
) -> None:
    """
    Render the app header.

    Args:
        title: Main title text
        subtitle: Subtitle text
        icon: Emoji icon
    """
    st.markdown(f"""
        <h1 style='text-align: center; color: #667eea;'>
            {icon} {title}
        </h1>
        <p style='text-align: center; color: #999;'>
            {subtitle}
        </p>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the app footer."""
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #666;'>
            Enhanced Multi-Modal Person Analysis System v3.0 |
            Powered with ❤️ by YOLOv8, DeepSORT, DeepFace & Advanced Analytics
        </p>
    """, unsafe_allow_html=True)


def badge(text: str, badge_type: str = "success") -> str:
    """
    Create an HTML badge.

    Args:
        text: Badge text
        badge_type: 'success', 'warning', or 'danger'

    Returns:
        HTML string for the badge
    """
    return f'<span class="{badge_type}-badge">{text}</span>'
