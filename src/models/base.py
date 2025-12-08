"""
Abstract base class for all ML models.
Defines the interface that all model wrappers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for ML model wrappers.

    All model classes should inherit from this and implement
    the required methods for consistent interface across the system.
    """

    def __init__(self):
        self._model: Optional[Any] = None
        self._is_loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights and initialize the model.

        This method should be called before any predictions.
        It should handle downloading weights if necessary.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> Any:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array (BGR format)
            **kwargs: Additional model-specific parameters

        Returns:
            Model-specific prediction results
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded, loading it if necessary."""
        if not self._is_loaded:
            self.load()

    def unload(self) -> None:
        """
        Unload the model to free memory.

        Override this method if the model requires special cleanup.
        """
        self._model = None
        self._is_loaded = False

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.__class__.__name__}({status})"
