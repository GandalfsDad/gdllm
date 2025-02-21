from abc import ABC, abstractmethod

class AbstractMessage(ABC):
    @abstractmethod
    def print(self):
        pass