"""
Alert system for real-time notifications.
Extracted from app4.py lines 575-646.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from src.config.settings import AlertSettings


@dataclass
class Alert:
    """Represents a single alert."""
    type: str
    severity: str  # 'high', 'medium', 'low'
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%H:%M:%S'))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'type': self.type,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp,
            'details': self.details
        }


class AlertSystem:
    """
    Real-time alert generation system.

    Monitors various conditions and generates alerts
    when thresholds are exceeded.
    """

    ALERT_TYPES = [
        'crowding',
        'loitering',
        'social_distancing',
        'unusual_behavior',
        'queue_overflow'
    ]

    def __init__(self, config: Optional[AlertSettings] = None):
        """
        Initialize the alert system.

        Args:
            config: Alert settings configuration
        """
        self.config = config or AlertSettings()
        self.alerts: List[Alert] = []

        self.rules = {
            'crowding': {
                'threshold': self.config.crowding_threshold,
                'enabled': self.config.crowding_enabled
            },
            'loitering': {
                'threshold': self.config.loitering_duration,
                'enabled': self.config.loitering_enabled
            },
            'social_distancing': {
                'threshold': self.config.min_social_distance,
                'enabled': self.config.social_distancing_enabled
            },
            'unusual_behavior': {
                'enabled': self.config.unusual_behavior_enabled
            }
        }

    def check_crowding(self, person_count: int) -> Optional[Alert]:
        """
        Check for crowding condition.

        Args:
            person_count: Current number of detected persons

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if not self.rules['crowding']['enabled']:
            return None

        threshold = self.rules['crowding']['threshold']

        if person_count > threshold:
            alert = Alert(
                type='crowding',
                severity='high',
                message=f'HIGH CROWD DENSITY: {person_count} persons detected',
                details={'count': person_count, 'threshold': threshold}
            )
            self.alerts.append(alert)
            return alert

        return None

    def check_loitering(
        self,
        trajectory_analyzer: Any,
        min_duration: Optional[float] = None
    ) -> List[Alert]:
        """
        Check for loitering conditions.

        Args:
            trajectory_analyzer: TrajectoryAnalyzer instance
            min_duration: Override minimum duration threshold

        Returns:
            List of loitering alerts
        """
        if not self.rules['loitering']['enabled']:
            return []

        threshold = min_duration or self.rules['loitering']['threshold']
        alerts = []

        for track_id in trajectory_analyzer.trajectories:
            if trajectory_analyzer.detect_loitering(track_id):
                dwell_time = trajectory_analyzer.get_dwell_time(track_id)

                if dwell_time > threshold:
                    alert = Alert(
                        type='loitering',
                        severity='medium',
                        message=f'LOITERING DETECTED: Person ID {track_id} (Duration: {dwell_time:.1f}s)',
                        details={
                            'track_id': track_id,
                            'duration': dwell_time,
                            'threshold': threshold
                        }
                    )
                    alerts.append(alert)
                    self.alerts.append(alert)

        return alerts

    def check_social_distancing(self, violations_count: int) -> Optional[Alert]:
        """
        Check for social distancing violations.

        Args:
            violations_count: Number of current violations

        Returns:
            Alert if violations exist, None otherwise
        """
        if not self.rules['social_distancing']['enabled']:
            return None

        if violations_count > 0:
            severity = 'high' if violations_count >= 5 else 'medium'

            alert = Alert(
                type='social_distancing',
                severity=severity,
                message=f'SOCIAL DISTANCING VIOLATION: {violations_count} pairs too close',
                details={'violations': violations_count}
            )
            self.alerts.append(alert)
            return alert

        return None

    def check_unusual_behavior(
        self,
        behavior_analyzer: Any
    ) -> List[Alert]:
        """
        Check for unusual behavior alerts.

        Args:
            behavior_analyzer: BehaviorPatternAnalyzer instance

        Returns:
            List of behavior alerts
        """
        if not self.rules['unusual_behavior']['enabled']:
            return []

        alerts = []
        recent_anomalies = behavior_analyzer.get_anomalies(limit=5)

        for anomaly in recent_anomalies:
            # Skip if already alerted (check by track_id and timestamp proximity)
            alert = Alert(
                type='unusual_behavior',
                severity=anomaly['severity'],
                message=f'UNUSUAL BEHAVIOR: Person ID {anomaly["track_id"]} - {anomaly["behavior"]}',
                details=anomaly
            )
            alerts.append(alert)
            self.alerts.append(alert)

        return alerts

    def check_queue_overflow(
        self,
        queue_system: Any,
        max_length: int = 10
    ) -> List[Alert]:
        """
        Check for queue overflow conditions.

        Args:
            queue_system: QueueManagementSystem instance
            max_length: Maximum allowed queue length

        Returns:
            List of queue overflow alerts
        """
        alerts = []
        stats = queue_system.get_statistics()

        for queue_name, queue_stats in stats.items():
            if queue_stats['current_count'] > max_length:
                alert = Alert(
                    type='queue_overflow',
                    severity='medium',
                    message=f'QUEUE OVERFLOW: {queue_name} has {queue_stats["current_count"]} persons',
                    details={
                        'queue_name': queue_name,
                        'count': queue_stats['current_count'],
                        'max_length': max_length
                    }
                )
                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def get_recent_alerts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts as dictionaries
        """
        return [alert.to_dict() for alert in self.alerts[-limit:][::-1]]

    def get_alerts_by_type(self, alert_type: str) -> List[Dict[str, Any]]:
        """
        Get alerts filtered by type.

        Args:
            alert_type: Type of alerts to retrieve

        Returns:
            List of matching alerts
        """
        return [
            alert.to_dict()
            for alert in self.alerts
            if alert.type == alert_type
        ]

    def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get alerts filtered by severity.

        Args:
            severity: Severity level ('high', 'medium', 'low')

        Returns:
            List of matching alerts
        """
        return [
            alert.to_dict()
            for alert in self.alerts
            if alert.severity == severity
        ]

    def get_alert_counts(self) -> Dict[str, int]:
        """Get count of alerts by type."""
        counts = {t: 0 for t in self.ALERT_TYPES}
        for alert in self.alerts:
            if alert.type in counts:
                counts[alert.type] += 1
        return counts

    def set_rule(self, rule_name: str, **kwargs) -> None:
        """
        Update an alert rule.

        Args:
            rule_name: Name of the rule to update
            **kwargs: Rule parameters to update
        """
        if rule_name in self.rules:
            self.rules[rule_name].update(kwargs)

    def enable_rule(self, rule_name: str) -> None:
        """Enable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name]['enabled'] = True

    def disable_rule(self, rule_name: str) -> None:
        """Disable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name]['enabled'] = False

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []

    def export_alerts(self) -> List[Dict[str, Any]]:
        """Export all alerts as list of dictionaries."""
        return [alert.to_dict() for alert in self.alerts]
