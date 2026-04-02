from abc import ABC, abstractmethod


class ForceEffect(ABC):
    @abstractmethod
    def apply(self):
        pass
