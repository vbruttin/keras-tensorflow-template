from abc import ABC, abstractmethod

from utils.config import Config


class BaseDataLoader(ABC):

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    # @abstractmethod
    # def data_generator(self):
    #     pass
