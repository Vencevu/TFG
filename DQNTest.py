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

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(self.sess)
        # self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer('self.log_dir')

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.graph = tf.compat.v1.get_default_graph()
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            self.model = model
        self._train_dir = self.log_dir + '\\train'

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=Config.REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{Config.MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        
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

    def update_replay_memory(self, transistion):
        self.replay_memory.append(transistion)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < Config.MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, Config.MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, Config.PREDICTION_BATCH_SIZE)
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, Config.PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + Config.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q  ## Q para la acción que tomamos es ahora igual al nuevo valor de Q

            X.append(current_state)  ## La imagen que tenemos
            y.append(current_qs)  ## la Q que tenemos
        
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=Config.TRAINING_BATCH_SIZE, verbose=0,
                        shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > Config.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
