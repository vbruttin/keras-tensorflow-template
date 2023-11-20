from abc import ABC, abstractmethod

from utils.config import Config
from utils.logger import Logger
from .base_data_loader import BaseDataLoader
from .base_model import BaseModel

logger = Logger()


class BaseTrain(ABC):

    def __init__(self, model: BaseModel, data_train: BaseDataLoader, data_validate: BaseDataLoader, config: Config):
        self.model = model
        self.train_data = data_train
        self.val_data = data_validate
        self.config = config

    @abstractmethod
    def train(self):
        pass
