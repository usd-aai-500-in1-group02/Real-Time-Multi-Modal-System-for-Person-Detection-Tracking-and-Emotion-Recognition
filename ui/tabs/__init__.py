"""UI Tab modules."""

from ui.tabs.upload import render_upload_tab
from ui.tabs.webcam import render_webcam_tab
from ui.tabs.analytics import render_analytics_tab
from ui.tabs.advanced import render_advanced_tab
from ui.tabs.reports import render_reports_tab

__all__ = [
    "render_upload_tab",
    "render_webcam_tab",
    "render_analytics_tab",
    "render_advanced_tab",
    "render_reports_tab",
]
