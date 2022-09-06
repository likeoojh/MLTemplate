from abc import *


class NoBrainnerModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        return

    @abstractmethod
    def fit(self):
        return
