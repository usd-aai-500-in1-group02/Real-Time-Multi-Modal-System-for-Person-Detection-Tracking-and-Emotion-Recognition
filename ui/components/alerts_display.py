"""
Alert display components.
"""

from typing import Dict, Any, List
import streamlit as st
import pandas as pd


def render_alert_box(alert: Dict[str, Any]) -> None:
    """
    Render a single alert box.

    Args:
        alert: Alert dictionary with type, severity, message, timestamp
    """
    severity = alert.get('severity', 'low')
    severity_class = f"alert-{severity}"

    st.markdown(f"""
        <div class="alert-box {severity_class}">
            {alert.get('message', 'Alert')}<br>
            <small>Time: {alert.get('timestamp', 'Unknown')}</small>
        </div>
    """, unsafe_allow_html=True)


def render_alerts(alerts: List[Dict[str, Any]], title: str = "Recent Alerts") -> None:
    """
    Render a list of alerts.

    Args:
        alerts: List of alert dictionaries
        title: Section title
    """
    if not alerts:
        return

    st.subheader(f"🚨 {title}")

    for alert in alerts:
        render_alert_box(alert)


def render_alert_summary(alert_counts: Dict[str, int]) -> None:
    """
    Render alert count summary.

    Args:
        alert_counts: Dictionary of alert_type: count
    """
    if not alert_counts or all(v == 0 for v in alert_counts.values()):
        st.success("No alerts triggered")
        return

    cols = st.columns(len(alert_counts))

    type_icons = {
        'crowding': '👥',
        'loitering': '🚶',
        'social_distancing': '📏',
        'unusual_behavior': '⚠️',
        'queue_overflow': '📋'
    }

    for i, (alert_type, count) in enumerate(alert_counts.items()):
        with cols[i]:
            icon = type_icons.get(alert_type, '🔔')
            label = alert_type.replace('_', ' ').title()

            if count > 0:
                st.metric(f"{icon} {label}", count)
            else:
                st.metric(f"{icon} {label}", count, delta_color="off")


def render_alert_table(alerts: List[Dict[str, Any]]) -> None:
    """
    Render alerts as a table.

    Args:
        alerts: List of alert dictionaries
    """
    if not alerts:
        st.info("No alerts recorded yet")
        return

    df = pd.DataFrame(alerts)

    # Reorder columns if they exist
    desired_order = ['timestamp', 'type', 'severity', 'message']
    existing_cols = [col for col in desired_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in desired_order]
    df = df[existing_cols + other_cols]

    # Style severity column
    def color_severity(val):
        colors = {
            'high': 'background-color: rgba(255, 0, 0, 0.3)',
            'medium': 'background-color: rgba(255, 165, 0, 0.3)',
            'low': 'background-color: rgba(255, 255, 0, 0.3)'
        }
        return colors.get(val, '')

    if 'severity' in df.columns:
        styled_df = df.style.applymap(
            color_severity,
            subset=['severity']
        )
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def render_alert_filters() -> Dict[str, Any]:
    """
    Render alert filter controls.

    Returns:
        Dictionary of filter settings
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['high', 'medium', 'low'],
            default=['high', 'medium', 'low']
        )

    with col2:
        type_filter = st.multiselect(
            "Filter by Type",
            options=['crowding', 'loitering', 'social_distancing', 'unusual_behavior'],
            default=['crowding', 'loitering', 'social_distancing', 'unusual_behavior']
        )

    with col3:
        limit = st.number_input(
            "Max Alerts",
            min_value=5,
            max_value=100,
            value=20
        )

    return {
        'severity': severity_filter,
        'type': type_filter,
        'limit': limit
    }


def filter_alerts(
    alerts: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter alerts based on criteria.

    Args:
        alerts: List of alerts
        filters: Filter settings from render_alert_filters()

    Returns:
        Filtered list of alerts
    """
    filtered = []

    for alert in alerts:
        if alert.get('severity') not in filters.get('severity', []):
            continue
        if alert.get('type') not in filters.get('type', []):
            continue
        filtered.append(alert)

    limit = filters.get('limit', 20)
    return filtered[:limit]
