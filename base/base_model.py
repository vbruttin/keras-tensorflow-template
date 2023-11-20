from abc import ABC, abstractmethod
from pathlib import Path

from utils.config import Config
from utils.logger import Logger

logger = Logger()


class BaseModel(ABC):

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def save(self, checkpoint_path: Path):
        if self.model is None:
            raise Exception("Model has not been built. Cannot save.")

        logger.info("Saving model...")
        self.model.save_weights(checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

    def load(self, checkpoint_path: Path):
        if self.model is None:
            raise Exception("Model has not been built. Cannot load.")

        logger.info(f"Loading model checkpoint {checkpoint_path} ..")
        self.model.load_weights(checkpoint_path)
        logger.info("Model loaded")

    @abstractmethod
    def build_model(self):
        pass
