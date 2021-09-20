from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np


class Layer(ABC):
    """
    Returns weights tensor if layer is trainable.
    Returns None for non-trainable layers.
    """
    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return None

    """
    Returns bias tensor if layer is trainable.
    Returns None for non-trainable layers.
    """
    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        return None

    """
    Perform layer forward propagation logic.
    """
    @abstractmethod
    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        pass
