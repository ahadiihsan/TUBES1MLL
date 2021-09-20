from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np


class Layer(ABC):
    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return None

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        return None

    @abstractmethod
    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        pass
