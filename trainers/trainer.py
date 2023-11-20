import os
from pathlib import Path

from keras.callbacks import ModelCheckpoint, TensorBoard

from base.base_data_loader import BaseDataLoader
from base.base_model import BaseModel
from base.base_trainer import BaseTrain
from utils.config import Config
from utils.logger import Logger
from utils.utils import save_as_pickle

logger = Logger()


class ModelTrainer(BaseTrain):

    def __init__(self, model: BaseModel, data_train: BaseDataLoader, data_validate: BaseDataLoader, config: Config):
        super().__init__(model, data_train, data_validate, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.extend([
            ModelCheckpoint(filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                                  f'{self.config.exp.name}-{{epoch:02d}}-{{val_acc:.4f}}.h5'),
                            monitor=self.config.callbacks.checkpoint_monitor,
                            mode=self.config.callbacks.checkpoint_mode,
                            save_best_only=self.config.callbacks.checkpoint_save_best_only,
                            verbose=self.config.callbacks.checkpoint_verbose,
                            save_format="h5"),
            TensorBoard(log_dir=self.config.callbacks.tensorboard_dir)
        ])

    def train(self):
        self.train_x, self.train_y = self.train_data
        self.val_x, self.val_y = self.val_data

        history = self.model.model.fit(
            self.train_x.values,
            self.train_y.values,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=(self.val_x.values, self.val_y.values),
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

        #show model summary
        logger.info(self.model.model.summary())
        #save train and validation metric lists as pickle files
        if self.config.trainer.save_pickle:
            checkpoint_dir = Path(self.config.callbacks.checkpoint_dir)
            logger.info('Saving training..')
            save_as_pickle(self.acc, checkpoint_dir / 'train_acc.list')
            logger.info('Saving validation..')
            save_as_pickle(self.val_acc, checkpoint_dir / 'val_acc.list')
