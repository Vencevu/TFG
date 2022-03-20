import time
import Config
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Flatten

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        
    def create_model(self):
        ## input: RGB data, should be normalized when coming into CNN
        base_model = Xception(weights=None, include_top=False, input_shape=(Config.IM_HEIGHT, Config.IM_WIDTH, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        predictions = Dense(3, activation="linear")(x)  ## output layer include three nuros, representing three actions
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
        return model
