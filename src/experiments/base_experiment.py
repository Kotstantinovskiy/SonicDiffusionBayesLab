from abc import ABC, abstractmethod


class BaseMethod(ABC):
    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def setup_dataset(self):
        pass

    @abstractmethod
    def run_experiment(self):
        pass
