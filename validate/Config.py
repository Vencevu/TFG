FPS = 60
IM_WIDTH = 320
IM_HEIGHT = 240

N_ACTIONS = 4

START_X = -30
START_Y = 26 # 26 70 140 -60

GOAL_X = 0
GOAL_Y = 26 # 26 70 140 -60

SECONDS_PER_EPISODE = 10

REPLAY_MEMORY_SIZE = 5_000 #5_000 # 10_000 15_000
MIN_REPLAY_MEMORY_SIZE = 1_000

MINIBATCH_SIZE = 64 #32 #64
PREDICTION_BATCH_SIZE = 1

TRAINING_BATCH_SIZE = 32
UPDATE_TARGET_EVERY = 5

MODEL_NAME = "Xception"
MEMORY_FRACTION = 0.8

MIN_REWARD = -200
MAX_REWARD = 200
INT_REWARD = 100

EPISODES = 500

DISCOUNT = 0.59
epsilon = 0.4168
EPSILON_DECAY = 0.99975  ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10  ## checking per 10 episodes
SHOW_PREVIEW = False  ## for debugging purpose
SHOW_CAM = False

MODEL_PATH = 'models/Weights_RL_Model.h5'