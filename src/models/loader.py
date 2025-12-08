"""
Model loader factory with caching support.
"""

from typing import Dict, Optional, Any
import streamlit as st

from src.config.settings import AppConfig, ModelSettings, TrackerSettings
from src.models.detector import PersonDetector
from src.models.segmenter import PersonSegmenter
from src.models.tracker import PersonTracker
from src.models.face_detector import FaceDetector
from src.models.emotion import EmotionRecognizer


class ModelLoader:
    """
    Factory for loading and caching ML models.

    Provides centralized model management with lazy loading
    and optional Streamlit caching.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the model loader.

        Args:
            config: Application configuration
        """
        self.config = config or AppConfig.default()
        self._models: Dict[str, Any] = {}

    def get_detector(self) -> PersonDetector:
        """
        Get the person detector model.

        Returns:
            Loaded PersonDetector instance
        """
        if 'detector' not in self._models:
            detector = PersonDetector(self.config.model)
            detector.load()
            self._models['detector'] = detector
        return self._models['detector']

    def get_segmenter(self) -> PersonSegmenter:
        """
        Get the person segmenter model.

        Returns:
            Loaded PersonSegmenter instance
        """
        if 'segmenter' not in self._models:
            segmenter = PersonSegmenter(self.config.model)
            segmenter.load()
            self._models['segmenter'] = segmenter
        return self._models['segmenter']

    def get_tracker(self) -> PersonTracker:
        """
        Get the person tracker.

        Returns:
            Loaded PersonTracker instance
        """
        if 'tracker' not in self._models:
            tracker = PersonTracker(self.config.tracker)
            tracker.load()
            self._models['tracker'] = tracker
        return self._models['tracker']

    def get_face_detector(self) -> FaceDetector:
        """
        Get the face detector model.

        Returns:
            Loaded FaceDetector instance
        """
        if 'face_detector' not in self._models:
            face_detector = FaceDetector(self.config.model)
            face_detector.load()
            self._models['face_detector'] = face_detector
        return self._models['face_detector']

    def get_emotion_recognizer(self) -> EmotionRecognizer:
        """
        Get the emotion recognizer model.

        Returns:
            Loaded EmotionRecognizer instance
        """
        if 'emotion' not in self._models:
            emotion = EmotionRecognizer(self.config.model)
            emotion.load()
            self._models['emotion'] = emotion
        return self._models['emotion']

    def get_all_models(self) -> Dict[str, Any]:
        """
        Load and return all models.

        Returns:
            Dictionary of all loaded models
        """
        return {
            'detector': self.get_detector(),
            'segmenter': self.get_segmenter(),
            'tracker': self.get_tracker(),
            'face_detector': self.get_face_detector(),
            'emotion': self.get_emotion_recognizer()
        }

    def get_core_models(self) -> Dict[str, Any]:
        """
        Load and return only core models (detector, tracker).

        Returns:
            Dictionary of core models
        """
        return {
            'detector': self.get_detector(),
            'tracker': self.get_tracker()
        }

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a specific model is loaded.

        Args:
            model_name: Name of the model

        Returns:
            True if model is loaded
        """
        return model_name in self._models and self._models[model_name].is_loaded

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        for model in self._models.values():
            if hasattr(model, 'unload'):
                model.unload()
        self._models.clear()


@st.cache_resource
def load_models_cached(config_dict: Optional[Dict] = None) -> ModelLoader:
    """
    Load models with Streamlit caching.

    This function is cached by Streamlit to avoid reloading
    models on every rerun.

    Args:
        config_dict: Optional configuration dictionary

    Returns:
        ModelLoader instance with loaded models
    """
    if config_dict:
        config = AppConfig.from_dict(config_dict)
    else:
        config = AppConfig.default()

    loader = ModelLoader(config)

    # Pre-load core models
    loader.get_detector()
    loader.get_tracker()

    return loader


def create_model_loader(config: Optional[AppConfig] = None) -> ModelLoader:
    """
    Create a new model loader instance.

    Use this for non-Streamlit contexts or when you need
    a fresh loader without caching.

    Args:
        config: Application configuration

    Returns:
        New ModelLoader instance
    """
    return ModelLoader(config)
