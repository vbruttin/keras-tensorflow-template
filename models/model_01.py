from keras import Sequential
from keras.layers import Dense

from base.base_model import BaseModel
from utils.config import Config


class Model(BaseModel):

    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()

        self.model.add(Dense(self.config.model.first_layers_dense, activation='relu'))

        #middle layers number
        for _ in range(self.config.model.midlayer_num):
            self.model.add(Dense(self.config.model.middle_layers_dense, activation='relu'))

        #last layer
        self.model.add(Dense(self.config.model.last_layer_dense, activation=self.config.model.last_activation))

        self.model.compile(loss=self.config.model.loss,
                           optimizer=self.config.model.optimizer,
                           metrics=['acc'])
