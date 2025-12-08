"""
Performance metrics evaluator.
NEW: Implements missing PDF requirement for performance metrics
(mAP, precision, recall, tracking accuracy, emotion classification accuracy).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ap: float = 0.0  # Average Precision
    map50: float = 0.0  # mAP at IoU=0.50
    map75: float = 0.0  # mAP at IoU=0.75
    map: float = 0.0  # mAP at IoU=0.50:0.95

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_predictions: int = 0
    total_ground_truth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'ap': self.ap,
            'mAP@50': self.map50,
            'mAP@75': self.map75,
            'mAP@50:95': self.map,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }


@dataclass
class TrackingMetrics:
    """Multi-object tracking evaluation metrics."""
    mota: float = 0.0  # Multiple Object Tracking Accuracy
    motp: float = 0.0  # Multiple Object Tracking Precision
    idf1: float = 0.0  # ID F1 Score
    id_switches: int = 0
    fragmentations: int = 0
    mostly_tracked: int = 0
    mostly_lost: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'MOTA': self.mota,
            'MOTP': self.motp,
            'IDF1': self.idf1,
            'ID_Switches': self.id_switches,
            'Fragmentations': self.fragmentations,
            'Mostly_Tracked': self.mostly_tracked,
            'Mostly_Lost': self.mostly_lost,
            'FP': self.false_positives,
            'FN': self.false_negatives
        }


@dataclass
class EmotionMetrics:
    """Emotion classification evaluation metrics."""
    accuracy: float = 0.0
    precision_per_class: Dict[str, float] = field(default_factory=dict)
    recall_per_class: Dict[str, float] = field(default_factory=dict)
    f1_per_class: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision_per_class': self.precision_per_class,
            'recall_per_class': self.recall_per_class,
            'f1_per_class': self.f1_per_class
        }


class MetricsEvaluator:
    """
    Evaluator for computing performance metrics.

    Provides methods to calculate:
    - Detection metrics (precision, recall, mAP)
    - Tracking metrics (MOTA, MOTP, IDF1)
    - Emotion classification metrics (accuracy, per-class metrics)
    """

    EMOTION_CLASSES = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
    IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

    def __init__(self):
        """Initialize the metrics evaluator."""
        self._detection_results: List[Dict] = []
        self._tracking_results: List[Dict] = []
        self._emotion_results: List[Dict] = []

    def calculate_iou(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_precision_recall(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            predictions: List of {'bbox': [x1,y1,x2,y2], 'confidence': float}
            ground_truth: List of {'bbox': [x1,y1,x2,y2]}
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not predictions:
            return (0.0, 0.0, 0.0) if ground_truth else (1.0, 1.0, 1.0)

        if not ground_truth:
            return (0.0, 0.0, 0.0)

        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)

        matched_gt = set()
        tp, fp = 0, 0

        for pred in predictions:
            pred_bbox = np.array(pred['bbox'])
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                gt_bbox = np.array(gt['bbox'])
                iou = self.calculate_iou(pred_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truth) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return (precision, recall, f1)

    def calculate_ap(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Calculate Average Precision at a specific IoU threshold.

        Args:
            predictions: List of predictions with confidence
            ground_truth: List of ground truth boxes
            iou_threshold: IoU threshold

        Returns:
            Average Precision value
        """
        if not ground_truth:
            return 1.0 if not predictions else 0.0

        if not predictions:
            return 0.0

        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        matched_gt = set()

        for i, pred in enumerate(predictions):
            pred_bbox = np.array(pred['bbox'])
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                gt_bbox = np.array(gt['bbox'])
                iou = self.calculate_iou(pred_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1

        # Calculate cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(ground_truth)

        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if np.any(mask):
                ap += np.max(precision[mask])
        ap /= 11

        return ap

    def calculate_map(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truth: List[List[Dict]],
        iou_thresholds: Optional[List[float]] = None
    ) -> DetectionMetrics:
        """
        Calculate mean Average Precision across multiple images.

        Args:
            all_predictions: List of predictions per image
            all_ground_truth: List of ground truth per image
            iou_thresholds: List of IoU thresholds (default: 0.5 to 0.95)

        Returns:
            DetectionMetrics object
        """
        if iou_thresholds is None:
            iou_thresholds = list(self.IOU_THRESHOLDS)

        metrics = DetectionMetrics()

        # Flatten all predictions and ground truth
        all_preds_flat = []
        all_gt_flat = []
        total_tp, total_fp, total_fn = 0, 0, 0

        for preds, gts in zip(all_predictions, all_ground_truth):
            all_preds_flat.extend(preds)
            all_gt_flat.extend(gts)

            p, r, f1 = self.calculate_precision_recall(preds, gts, iou_threshold=0.5)
            matched = int(r * len(gts))
            total_tp += matched
            total_fp += len(preds) - matched
            total_fn += len(gts) - matched

        # Calculate mAP at different IoU thresholds
        aps = []
        for iou_thresh in iou_thresholds:
            ap_per_image = []
            for preds, gts in zip(all_predictions, all_ground_truth):
                ap = self.calculate_ap(preds, gts, iou_thresh)
                ap_per_image.append(ap)
            aps.append(np.mean(ap_per_image) if ap_per_image else 0)

        # Calculate metrics
        metrics.map50 = aps[0] if aps else 0  # mAP at IoU=0.50
        metrics.map75 = aps[5] if len(aps) > 5 else 0  # mAP at IoU=0.75
        metrics.map = np.mean(aps) if aps else 0  # mAP at IoU=0.50:0.95
        metrics.ap = metrics.map50

        metrics.true_positives = total_tp
        metrics.false_positives = total_fp
        metrics.false_negatives = total_fn
        metrics.total_predictions = total_tp + total_fp
        metrics.total_ground_truth = total_tp + total_fn

        metrics.precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        metrics.recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        metrics.f1_score = (
            2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            if (metrics.precision + metrics.recall) > 0 else 0
        )

        return metrics

    def calculate_tracking_metrics(
        self,
        predicted_tracks: List[Dict],
        ground_truth_tracks: List[Dict],
        iou_threshold: float = 0.5
    ) -> TrackingMetrics:
        """
        Calculate multi-object tracking metrics.

        Args:
            predicted_tracks: List of predicted track dictionaries
            ground_truth_tracks: List of ground truth track dictionaries
            iou_threshold: IoU threshold for matching

        Returns:
            TrackingMetrics object
        """
        metrics = TrackingMetrics()

        # Simplified MOTA calculation
        total_gt = len(ground_truth_tracks)
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_matches = 0
        total_iou = 0

        if total_gt == 0:
            return metrics

        # Track matching across frames
        matched_ids = {}

        for pred in predicted_tracks:
            pred_bbox = np.array(pred.get('bbox', [0, 0, 0, 0]))
            pred_id = pred.get('track_id', -1)
            best_iou = 0
            best_gt_id = None

            for gt in ground_truth_tracks:
                gt_bbox = np.array(gt.get('bbox', [0, 0, 0, 0]))
                gt_id = gt.get('track_id', -1)

                iou = self.calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_gt_id is not None:
                total_matches += 1
                total_iou += best_iou

                # Check for ID switch
                if best_gt_id in matched_ids and matched_ids[best_gt_id] != pred_id:
                    total_id_switches += 1
                matched_ids[best_gt_id] = pred_id
            else:
                total_fp += 1

        total_fn = total_gt - total_matches

        # Calculate MOTA
        metrics.mota = 1 - (total_fn + total_fp + total_id_switches) / total_gt if total_gt > 0 else 0
        metrics.mota = max(0, min(1, metrics.mota))

        # Calculate MOTP
        metrics.motp = total_iou / total_matches if total_matches > 0 else 0

        metrics.id_switches = total_id_switches
        metrics.false_positives = total_fp
        metrics.false_negatives = total_fn

        return metrics

    def calculate_emotion_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        class_names: Optional[List[str]] = None
    ) -> EmotionMetrics:
        """
        Calculate emotion classification metrics.

        Args:
            predictions: List of predicted emotion labels
            ground_truth: List of ground truth emotion labels
            class_names: List of class names

        Returns:
            EmotionMetrics object
        """
        if class_names is None:
            class_names = self.EMOTION_CLASSES

        metrics = EmotionMetrics(class_names=class_names)

        if not predictions or not ground_truth:
            return metrics

        # Create confusion matrix
        n_classes = len(class_names)
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
        correct = 0

        for pred, gt in zip(predictions, ground_truth):
            if pred in class_to_idx and gt in class_to_idx:
                pred_idx = class_to_idx[pred]
                gt_idx = class_to_idx[gt]
                confusion[gt_idx, pred_idx] += 1

                if pred == gt:
                    correct += 1

        metrics.confusion_matrix = confusion
        metrics.accuracy = correct / len(predictions) if predictions else 0

        # Per-class metrics
        for i, class_name in enumerate(class_names):
            tp = confusion[i, i]
            fp = np.sum(confusion[:, i]) - tp
            fn = np.sum(confusion[i, :]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics.precision_per_class[class_name] = precision
            metrics.recall_per_class[class_name] = recall
            metrics.f1_per_class[class_name] = f1

        return metrics

    def evaluate_detection_session(
        self,
        frame_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate detection performance from a session.

        Args:
            frame_results: List of frame processing results

        Returns:
            Session evaluation metrics
        """
        total_detections = sum(r.get('person_count', 0) for r in frame_results)
        confidences = []

        for r in frame_results:
            confidences.extend(r.get('confidences', []))

        return {
            'total_frames': len(frame_results),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / len(frame_results) if frame_results else 0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0,
            'min_confidence': float(min(confidences)) if confidences else 0,
            'max_confidence': float(max(confidences)) if confidences else 0,
            'confidence_std': float(np.std(confidences)) if confidences else 0
        }

    def get_summary_report(
        self,
        detection_metrics: Optional[DetectionMetrics] = None,
        tracking_metrics: Optional[TrackingMetrics] = None,
        emotion_metrics: Optional[EmotionMetrics] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary report of all metrics.

        Args:
            detection_metrics: Detection evaluation results
            tracking_metrics: Tracking evaluation results
            emotion_metrics: Emotion classification results

        Returns:
            Summary report dictionary
        """
        report = {}

        if detection_metrics:
            report['detection'] = detection_metrics.to_dict()

        if tracking_metrics:
            report['tracking'] = tracking_metrics.to_dict()

        if emotion_metrics:
            report['emotion'] = emotion_metrics.to_dict()

        return report
