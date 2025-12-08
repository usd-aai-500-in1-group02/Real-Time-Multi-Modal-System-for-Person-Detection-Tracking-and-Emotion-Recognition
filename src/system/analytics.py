"""
Analytics dashboard for comprehensive data tracking.
Extracted from app4.py lines 648-718.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import json


class AnalyticsDashboard:
    """
    Comprehensive analytics tracking for the analysis session.

    Collects frame-by-frame metrics and provides summary
    statistics and export capabilities.
    """

    def __init__(self):
        """Initialize the analytics dashboard."""
        self.data = {
            'timestamps': [],
            'person_counts': [],
            'emotions': defaultdict(int),
            'detection_confidences': [],
            'processing_times': [],
            'frame_numbers': [],
            'distancing_violations': [],
            'behavior_anomalies': [],
            'fps_values': []
        }
        self.alerts: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self._last_frame_time = None

    def add_frame_data(
        self,
        frame_num: int,
        person_count: int,
        emotions: List[Optional[Dict[str, Any]]],
        confidences: List[float],
        proc_time: float,
        violations: int = 0,
        anomalies: int = 0
    ) -> None:
        """
        Add data for a processed frame.

        Args:
            frame_num: Frame number
            person_count: Number of detected persons
            emotions: List of emotion detection results
            confidences: List of detection confidence scores
            proc_time: Processing time in seconds
            violations: Number of distancing violations
            anomalies: Number of behavior anomalies
        """
        self.data['frame_numbers'].append(frame_num)
        self.data['timestamps'].append(datetime.now())
        self.data['person_counts'].append(person_count)
        self.data['detection_confidences'].extend(confidences)
        self.data['processing_times'].append(proc_time)
        self.data['distancing_violations'].append(violations)
        self.data['behavior_anomalies'].append(anomalies)

        # Calculate FPS
        if proc_time > 0:
            self.data['fps_values'].append(1.0 / proc_time)

        # Track emotions
        for emotion in emotions:
            if emotion is not None and 'emotion' in emotion:
                self.data['emotions'][emotion['emotion']] += 1

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the session.

        Returns:
            Dictionary with summary statistics
        """
        if not self.data['person_counts']:
            return {}

        person_counts = self.data['person_counts']
        processing_times = self.data['processing_times']
        confidences = self.data['detection_confidences']
        fps_values = self.data['fps_values']

        return {
            'total_frames': len(self.data['frame_numbers']),
            'avg_persons': float(np.mean(person_counts)),
            'max_persons': int(max(person_counts)),
            'min_persons': int(min(person_counts)),
            'std_persons': float(np.std(person_counts)),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'min_confidence': float(min(confidences)) if confidences else 0.0,
            'max_confidence': float(max(confidences)) if confidences else 0.0,
            'avg_processing_time': float(np.mean(processing_times)) if processing_times else 0.0,
            'avg_fps': float(np.mean(fps_values)) if fps_values else 0.0,
            'max_fps': float(max(fps_values)) if fps_values else 0.0,
            'min_fps': float(min(fps_values)) if fps_values else 0.0,
            'total_emotions': sum(self.data['emotions'].values()),
            'emotion_distribution': dict(self.data['emotions']),
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'total_distancing_violations': sum(self.data['distancing_violations']),
            'total_behavior_anomalies': sum(self.data['behavior_anomalies']),
            'avg_violations_per_frame': float(np.mean(self.data['distancing_violations'])) if self.data['distancing_violations'] else 0.0
        }

    def get_timeline_data(self) -> Dict[str, List]:
        """
        Get timeline data for plotting.

        Returns:
            Dictionary with timeline arrays
        """
        return {
            'frames': self.data['frame_numbers'].copy(),
            'person_counts': self.data['person_counts'].copy(),
            'processing_times': self.data['processing_times'].copy(),
            'violations': self.data['distancing_violations'].copy(),
            'anomalies': self.data['behavior_anomalies'].copy(),
            'fps': self.data['fps_values'].copy()
        }

    def get_emotion_data(self) -> Dict[str, int]:
        """Get emotion distribution data."""
        return dict(self.data['emotions'])

    def get_recent_frames(self, n: int = 10) -> Dict[str, List]:
        """
        Get data for the most recent N frames.

        Args:
            n: Number of recent frames

        Returns:
            Dictionary with recent frame data
        """
        return {
            'frames': self.data['frame_numbers'][-n:],
            'person_counts': self.data['person_counts'][-n:],
            'processing_times': self.data['processing_times'][-n:],
            'violations': self.data['distancing_violations'][-n:],
            'anomalies': self.data['behavior_anomalies'][-n:]
        }

    def export_to_csv(self) -> str:
        """
        Export data to CSV format.

        Returns:
            CSV string
        """
        df = pd.DataFrame({
            'Frame': self.data['frame_numbers'],
            'Timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in self.data['timestamps']],
            'Person_Count': self.data['person_counts'],
            'Processing_Time': self.data['processing_times'],
            'Distancing_Violations': self.data['distancing_violations'],
            'Behavior_Anomalies': self.data['behavior_anomalies']
        })
        return df.to_csv(index=False)

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export data to pandas DataFrame.

        Returns:
            DataFrame with analytics data
        """
        return pd.DataFrame({
            'Frame': self.data['frame_numbers'],
            'Timestamp': self.data['timestamps'],
            'Person_Count': self.data['person_counts'],
            'Processing_Time': self.data['processing_times'],
            'Distancing_Violations': self.data['distancing_violations'],
            'Behavior_Anomalies': self.data['behavior_anomalies']
        })

    def export_to_json(self) -> str:
        """
        Export full report to JSON format.

        Returns:
            JSON string
        """
        report = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.session_start).total_seconds()
            },
            'summary': self.get_summary_stats(),
            'emotions': dict(self.data['emotions']),
            'timeline': {
                'frames': self.data['frame_numbers'],
                'person_counts': self.data['person_counts'],
                'violations': self.data['distancing_violations'],
                'anomalies': self.data['behavior_anomalies']
            }
        }
        return json.dumps(report, indent=2, default=str)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.data['processing_times']:
            return {}

        times = self.data['processing_times']
        fps = self.data['fps_values']

        return {
            'avg_processing_time_ms': float(np.mean(times) * 1000),
            'max_processing_time_ms': float(max(times) * 1000),
            'min_processing_time_ms': float(min(times) * 1000),
            'std_processing_time_ms': float(np.std(times) * 1000),
            'avg_fps': float(np.mean(fps)) if fps else 0.0,
            'fps_stability': float(np.std(fps)) if fps else 0.0,
            'total_processing_time': float(sum(times)),
            'frames_processed': len(times)
        }

    def reset(self) -> None:
        """Reset all analytics data."""
        self.data = {
            'timestamps': [],
            'person_counts': [],
            'emotions': defaultdict(int),
            'detection_confidences': [],
            'processing_times': [],
            'frame_numbers': [],
            'distancing_violations': [],
            'behavior_anomalies': [],
            'fps_values': []
        }
        self.alerts = []
        self.session_start = datetime.now()

    def add_alert(self, alert: Dict[str, Any]) -> None:
        """
        Add an alert to the analytics.

        Args:
            alert: Alert dictionary with type, severity, message, timestamp
        """
        self.alerts.append(alert)

    @property
    def frame_data(self) -> List[Dict[str, Any]]:
        """
        Get frame data in list-of-dicts format for compatibility.

        Returns:
            List of dictionaries, one per frame
        """
        frames = []
        for i in range(len(self.data['frame_numbers'])):
            frame_dict = {
                'frame_number': self.data['frame_numbers'][i],
                'person_count': self.data['person_counts'][i],
                'processing_time': self.data['processing_times'][i],
                'timestamp': self.data['timestamps'][i] if i < len(self.data['timestamps']) else None,
                'confidences': [],  # Individual frame confidences not stored separately
                'track_ids': [],  # Track IDs not stored in this structure
            }
            frames.append(frame_dict)
        return frames

    @property
    def emotion_counts(self) -> Dict[str, int]:
        """Get emotion counts dictionary."""
        return dict(self.data['emotions'])

    def get_statistics(self) -> Dict[str, Any]:
        """
        Alias for get_summary_stats for compatibility.

        Returns:
            Dictionary with summary statistics
        """
        return self.get_summary_stats()
