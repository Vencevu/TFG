#!/usr/bin/env python3
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import Config
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Flatten

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQNAgent:
    def __init__(self):

        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        ## main model (gets trained every step)
        # Un objeto Session encapsula el entorno en el que se ejecutan los objetos Operation
        # y se evalúan los objetos Tensor.

        self.sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(self.sess)

        # ## target model (this is what we .predict against every step)
        # self.target_model = self.create_model()
        # self.target_model.set_weights(self.model.get_weights())

        ## replay_memory se utiliza para recordar el tamaño de las acciones anteriores,
        # y luego ajustar nuestro modelo de esta cantidad de memoria haciendo un muestreo aleatorio
        self.replay_memory = deque(maxlen=Config.REPLAY_MEMORY_SIZE)  ## batch step
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{Config.MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0  # hará un seguimiento de cuándo es el momento de actualizar el modelo de destino
        self.graph = tf.compat.v1.get_default_graph()

        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            self.model = self.create_model()

            # self.model.save('/home/jarain78/Pycharm_Projects/AssistantRobotSimulator/DQN_PyBullet/models/RLModel/')
            # self.model.save_weights(
            #    '/home/jarain78/Pycharm_Projects/AssistantRobotSimulator/DQN_PyBullet/models/W_RLModel/')

            ## target model (this is what we .predict against every step)
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def save_rl_model(self):
        # self.model.save('models/rlmodel.h5')
        # self.save_model('models/', 'RL_Model', self.model)
        try:
            self.model.save_weights('models/Weights_RL_Model.h5')
        except:
            pass

        try:
            self.model.save('models/RL_Model.h5')
        except:
            pass

        try:
            self.model.save_model('models/RLModel')
        except:
            pass

    def create_model(self):
        ## input: RGB data, should be normalized when coming into CNN

        base_model = Xception(weights=None, include_top=False, input_shape=(Config.IM_HEIGHT, Config.IM_WIDTH, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        predictions = Dense(Config.N_ACTIONS, activation="linear")(x)  ## output layer include n nuros, representing n actions
        model = Model(inputs=base_model.input, outputs=predictions)
        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
        return model

    ## function handler
    # Añade los datos del paso a una matriz de repetición de memoria
    # (espacio de observación, acción, recompensa, nuevo espacio de observación, hecho)= (estado_actual, acción, recompensa, nuevo_estado, hecho)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):

        ## iniciar el entrenamiento sólo si ya se ha guardado cierto número de muestras.
        if len(self.replay_memory) < Config.MIN_REPLAY_MEMORY_SIZE:
            return

        ## si tenemos la cantidad adecuada de datos para entrenar, tenemos que seleccionar al azar
        # los datos que queremos entrenar de nuestra memoria
        minibatch = random.sample(self.replay_memory, Config.MINIBATCH_SIZE)

        ## obtener los estados actuales del minibatch y luego obtener los valores Q del modelo NN
        ## la transición se define así: transición = (estado_actual, acción, recompensa, nuevo_estado, hecho)
        current_states = np.array([transition[0] for transition in minibatch]) / 255

        ## This is the crazyly changed model:
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            current_qs_list = self.model.predict(current_states, Config.PREDICTION_BATCH_SIZE)

        ## This is normal model
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            future_qs_list = self.target_model.predict(new_current_states, Config.PREDICTION_BATCH_SIZE)

        ## datos de la imagen (datos RGB normalizados): entrada
        X = []
        ## acción que tomamos (valores Q): salida
        y = []

        ## calcular los valores de Q para el siguiente paso basado en la ecuación Qnew
        ## índice = paso
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + Config.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q  ## Q para la acción que tomamos es ahora igual al nuevo valor de Q

            X.append(current_state)  ## La imagen que tenemo
            y.append(current_qs)  ## la Q que tenemos

        ## only trying to log per episode, not actual training step, so we're going to use the below to keep track
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        ## fit our model
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            history = self.model.fit(np.array(X) / 255, np.array(y), batch_size=Config.TRAINING_BATCH_SIZE, verbose=0,
                           shuffle=False)
        
            ## updating to determine if we want to update target_model
            if log_this_step:
                self.target_update_counter += 1

            # If counter reaches set value, update target network with weights of main network
            if self.target_update_counter > Config.UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
            
        return history.history['accuracy'], history.history['loss']

    def get_qs(self, state):
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            q_out = self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
        return q_out

        ## primero para entrenar a algunas tonterías. sólo tiene que conseguir un ajuste quicl porque la primera
        # formación y la predicción es lento

    def train_in_loop(self):
        X = np.random.uniform(size=(1, Config.IM_HEIGHT, Config.IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


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
