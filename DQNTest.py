import numpy as np
import random
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Flatten


class DQNAgent:
    def __init__(self):

        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)