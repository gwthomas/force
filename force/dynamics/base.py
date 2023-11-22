from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    @abstractmethod
    def sample(self, states, actions):
        """
        Returns a sample of (s', r, d) given (s, a)
        """
        raise NotImplementedError