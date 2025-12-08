"""
Abstract base class for all analyzers.
Defines the interface that all analyzer classes must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseAnalyzer(ABC):
    """
    Abstract base class for analyzers.

    Analyzers process detection/tracking data to extract
    higher-level insights like heatmaps, trajectories, etc.
    """

    def __init__(self):
        self._is_initialized: bool = False

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the analyzer state with new data.

        This method is called for each frame/detection to
        accumulate data for analysis.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics from the analyzer.

        Returns:
            Dictionary containing analyzer-specific metrics
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the analyzer state.

        Clears all accumulated data and returns the
        analyzer to its initial state.
        """
        pass

    def draw(self, image: np.ndarray) -> np.ndarray:
        """
        Draw visualization on image.

        Override this method to add visual overlays.

        Args:
            image: Input image (BGR format)

        Returns:
            Image with visualization overlay
        """
        return image.copy()

    @property
    def is_initialized(self) -> bool:
        """Check if the analyzer has been initialized."""
        return self._is_initialized

    def __repr__(self) -> str:
        status = "initialized" if self._is_initialized else "not initialized"
        return f"{self.__class__.__name__}({status})"
