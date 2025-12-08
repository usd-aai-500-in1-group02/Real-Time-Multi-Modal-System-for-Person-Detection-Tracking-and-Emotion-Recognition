"""
Reports tab for generating and exporting analysis reports.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import streamlit as st
import pandas as pd

from src.system.analytics import AnalyticsDashboard


def render_reports_tab(
    analytics_dashboard: Optional[AnalyticsDashboard] = None
) -> None:
    """
    Render the reports tab for generating analysis reports.

    Args:
        analytics_dashboard: AnalyticsDashboard instance with collected data
    """
    st.header("📑 Reports & Export")

    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        options=[
            "Summary Report",
            "Detection Report",
            "Tracking Report",
            "Emotion Analysis Report",
            "Alert Report",
            "Full Analysis Report"
        ]
    )

    # Date range selection
    st.subheader("📅 Report Parameters")
    col1, col2 = st.columns(2)

    with col1:
        report_title = st.text_input("Report Title", value=f"{report_type} - {datetime.now().strftime('%Y-%m-%d')}")

    with col2:
        include_charts = st.checkbox("Include Charts", value=True)

    # Generate report button
    if st.button("📊 Generate Report", type="primary"):
        if analytics_dashboard is None or not analytics_dashboard.frame_data:
            st.warning("No data available for report generation. Process some images or videos first.")
        else:
            _generate_report(analytics_dashboard, report_type, report_title, include_charts)

    st.markdown("---")

    # Quick export options
    st.subheader("📥 Quick Export")
    _render_quick_export_options(analytics_dashboard)

    st.markdown("---")

    # Report history
    st.subheader("📚 Report History")
    _render_report_history()


def _generate_report(
    analytics: AnalyticsDashboard,
    report_type: str,
    title: str,
    include_charts: bool
) -> None:
    """Generate and display the selected report type."""
    st.markdown("---")
    st.subheader(f"📄 {title}")

    report_data = {
        'title': title,
        'type': report_type,
        'generated_at': datetime.now().isoformat(),
        'data': {}
    }

    if report_type == "Summary Report":
        report_data['data'] = _generate_summary_report(analytics, include_charts)
    elif report_type == "Detection Report":
        report_data['data'] = _generate_detection_report(analytics, include_charts)
    elif report_type == "Tracking Report":
        report_data['data'] = _generate_tracking_report(analytics, include_charts)
    elif report_type == "Emotion Analysis Report":
        report_data['data'] = _generate_emotion_report(analytics, include_charts)
    elif report_type == "Alert Report":
        report_data['data'] = _generate_alert_report(analytics)
    elif report_type == "Full Analysis Report":
        report_data['data'] = _generate_full_report(analytics, include_charts)

    # Store in session for history
    if 'report_history' not in st.session_state:
        st.session_state.report_history = []
    st.session_state.report_history.append(report_data)

    # Export buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        json_str = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{title.replace(' ', '_').lower()}.json",
            mime="application/json"
        )

    with col2:
        # Generate HTML report
        html_content = _generate_html_report(report_data)
        st.download_button(
            label="📥 Download HTML",
            data=html_content,
            file_name=f"{title.replace(' ', '_').lower()}.html",
            mime="text/html"
        )

    with col3:
        # Generate CSV if applicable
        if 'frame_data' in report_data['data']:
            df = pd.DataFrame(report_data['data']['frame_data'])
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_str,
                file_name=f"{title.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )


def _generate_summary_report(analytics: AnalyticsDashboard, include_charts: bool) -> Dict:
    """Generate summary report data."""
    stats = analytics.get_statistics()

    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Frames", stats.get('total_frames', 0))
    with col2:
        st.metric("Avg Persons", f"{stats.get('avg_persons', 0):.2f}")
    with col3:
        st.metric("Max Persons", stats.get('max_persons', 0))
    with col4:
        st.metric("Avg FPS", f"{stats.get('avg_fps', 0):.1f}")

    if include_charts and analytics.frame_data:
        st.markdown("### Person Count Trend")
        import plotly.express as px
        df = pd.DataFrame(analytics.frame_data)
        if 'frame_number' in df.columns and 'person_count' in df.columns:
            fig = px.line(df, x='frame_number', y='person_count', title='Person Count Over Time')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    return {
        'statistics': stats,
        'frame_count': len(analytics.frame_data)
    }


def _generate_detection_report(analytics: AnalyticsDashboard, include_charts: bool) -> Dict:
    """Generate detection report data."""
    st.markdown("### Detection Statistics")

    # Collect detection data
    total_detections = sum(d.get('person_count', 0) for d in analytics.frame_data)
    all_confidences = []
    for d in analytics.frame_data:
        all_confidences.extend(d.get('confidences', []))

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", total_detections)
    with col2:
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    with col3:
        st.metric("Frames Processed", len(analytics.frame_data))

    # Confidence distribution
    if include_charts and all_confidences:
        st.markdown("### Confidence Distribution")
        import plotly.express as px
        fig = px.histogram(x=all_confidences, nbins=20, title='Detection Confidence Distribution')
        fig.update_layout(template='plotly_dark', xaxis_title='Confidence', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    return {
        'total_detections': total_detections,
        'avg_confidence': avg_confidence,
        'confidence_values': all_confidences
    }


def _generate_tracking_report(analytics: AnalyticsDashboard, include_charts: bool) -> Dict:
    """Generate tracking report data."""
    st.markdown("### Tracking Statistics")

    # Collect tracking data
    all_track_ids = set()
    for d in analytics.frame_data:
        for tid in d.get('track_ids', []):
            if tid is not None:
                all_track_ids.add(tid)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Tracks", len(all_track_ids))
    with col2:
        st.metric("Frames with Tracking", len([d for d in analytics.frame_data if d.get('track_ids')]))
    with col3:
        avg_tracks = sum(len(d.get('track_ids', [])) for d in analytics.frame_data) / len(analytics.frame_data) if analytics.frame_data else 0
        st.metric("Avg Tracks/Frame", f"{avg_tracks:.1f}")

    return {
        'unique_tracks': len(all_track_ids),
        'track_ids': list(all_track_ids)
    }


def _generate_emotion_report(analytics: AnalyticsDashboard, include_charts: bool) -> Dict:
    """Generate emotion analysis report data."""
    st.markdown("### Emotion Analysis")

    emotion_counts = analytics.emotion_counts or {}

    if not emotion_counts or sum(emotion_counts.values()) == 0:
        st.info("No emotion data available")
        return {'emotion_counts': {}}

    # Display emotion distribution
    if include_charts:
        import plotly.express as px
        fig = px.pie(
            values=list(emotion_counts.values()),
            names=list(emotion_counts.keys()),
            title='Emotion Distribution',
            hole=0.3
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    # Emotion table
    st.markdown("### Emotion Counts")
    df = pd.DataFrame([
        {'Emotion': k, 'Count': v, 'Percentage': f"{v/sum(emotion_counts.values())*100:.1f}%"}
        for k, v in emotion_counts.items()
    ])
    st.dataframe(df, use_container_width=True)

    return {
        'emotion_counts': emotion_counts,
        'dominant_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    }


def _generate_alert_report(analytics: AnalyticsDashboard) -> Dict:
    """Generate alert report data."""
    st.markdown("### Alert Summary")

    alerts = analytics.alerts if hasattr(analytics, 'alerts') else []

    if not alerts:
        st.info("No alerts recorded")
        return {'alerts': [], 'total_alerts': 0}

    # Alert statistics
    alert_types = {}
    alert_severities = {}

    for alert in alerts:
        alert_type = alert.get('type', 'unknown')
        severity = alert.get('severity', 'low')
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        alert_severities[severity] = alert_severities.get(severity, 0) + 1

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By Type**")
        for atype, count in alert_types.items():
            st.metric(atype.replace('_', ' ').title(), count)

    with col2:
        st.markdown("**By Severity**")
        for sev, count in alert_severities.items():
            st.metric(sev.title(), count)

    # Recent alerts table
    st.markdown("### Recent Alerts")
    df = pd.DataFrame(alerts[-20:])
    st.dataframe(df, use_container_width=True)

    return {
        'alerts': alerts,
        'total_alerts': len(alerts),
        'by_type': alert_types,
        'by_severity': alert_severities
    }


def _generate_full_report(analytics: AnalyticsDashboard, include_charts: bool) -> Dict:
    """Generate comprehensive full report."""
    report = {}

    st.markdown("## 1. Summary")
    report['summary'] = _generate_summary_report(analytics, include_charts)

    st.markdown("---")
    st.markdown("## 2. Detection Analysis")
    report['detection'] = _generate_detection_report(analytics, include_charts)

    st.markdown("---")
    st.markdown("## 3. Tracking Analysis")
    report['tracking'] = _generate_tracking_report(analytics, include_charts)

    st.markdown("---")
    st.markdown("## 4. Emotion Analysis")
    report['emotion'] = _generate_emotion_report(analytics, include_charts)

    st.markdown("---")
    st.markdown("## 5. Alerts")
    report['alerts'] = _generate_alert_report(analytics)

    return report


def _render_quick_export_options(analytics: Optional[AnalyticsDashboard]) -> None:
    """Render quick export buttons."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Export Raw Data"):
            if analytics and analytics.frame_data:
                json_str = json.dumps(analytics.frame_data, indent=2, default=str)
                st.download_button(
                    "Download",
                    data=json_str,
                    file_name="raw_data.json",
                    mime="application/json"
                )
            else:
                st.warning("No data to export")

    with col2:
        if st.button("📈 Export Statistics"):
            if analytics:
                stats = analytics.get_statistics()
                json_str = json.dumps(stats, indent=2)
                st.download_button(
                    "Download",
                    data=json_str,
                    file_name="statistics.json",
                    mime="application/json"
                )
            else:
                st.warning("No statistics available")

    with col3:
        if st.button("😊 Export Emotions"):
            if analytics and analytics.emotion_counts:
                json_str = json.dumps(analytics.emotion_counts, indent=2)
                st.download_button(
                    "Download",
                    data=json_str,
                    file_name="emotions.json",
                    mime="application/json"
                )
            else:
                st.warning("No emotion data available")

    with col4:
        if st.button("⚠️ Export Alerts"):
            if analytics and hasattr(analytics, 'alerts') and analytics.alerts:
                json_str = json.dumps(analytics.alerts, indent=2, default=str)
                st.download_button(
                    "Download",
                    data=json_str,
                    file_name="alerts.json",
                    mime="application/json"
                )
            else:
                st.warning("No alerts to export")


def _render_report_history() -> None:
    """Render report generation history."""
    if 'report_history' not in st.session_state or not st.session_state.report_history:
        st.info("No reports generated yet in this session.")
        return

    for i, report in enumerate(reversed(st.session_state.report_history[-5:])):
        with st.expander(f"{report['title']} - {report['generated_at'][:19]}"):
            st.markdown(f"**Type:** {report['type']}")
            st.markdown(f"**Generated:** {report['generated_at']}")

            json_str = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="📥 Re-download",
                data=json_str,
                file_name=f"report_{i}.json",
                mime="application/json",
                key=f"redownload_{i}"
            )


def _generate_html_report(report_data: Dict) -> str:
    """Generate HTML version of report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_data['title']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #1e1e1e; color: #fff; }}
            h1 {{ color: #667eea; }}
            h2 {{ color: #00c853; border-bottom: 1px solid #333; padding-bottom: 10px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #2d2d2d; border-radius: 8px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
            .metric-label {{ font-size: 12px; color: #888; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ background: #2d2d2d; }}
            .timestamp {{ color: #888; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>{report_data['title']}</h1>
        <p class="timestamp">Generated: {report_data['generated_at']}</p>
        <p>Report Type: {report_data['type']}</p>

        <h2>Report Data</h2>
        <pre>{json.dumps(report_data['data'], indent=2, default=str)}</pre>
    </body>
    </html>
    """
    return html
