import numpy as np
import json

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import keras.backend as K
from tensorflow.python import debug as tf_debug
import argparse

INPUT_SHAPE = 4320
WINDOW_LENGTH = 4
ENV_NAME = "Backtest-v0"

dirname = "log-py/"
weights_filename = dirname+'dqn_{}_weights.h5f'.format(ENV_NAME)
checkpoint_weights_filename = dirname+'dqn_' + ENV_NAME + '_weights_{step}.h5f'
log_filename = dirname+'dqn_{}_log.json'.format(ENV_NAME)

parser = argparse.ArgumentParser()
parser.add_argument('--candle', type=str, default='data/bitflyerBTC_JPY_1min.csv')
parser.add_argument('--mode', choices=['train', 'train-load', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='Backtest-v0')
parser.add_argument('--weights', type=str, default=weights_filename)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

if args.debug:
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)


from backtest_env import BacktestEnv
import config
setting = config.get_setting()

# Get the environment and extract the number of actions.
env = BacktestEnv()
np.random.seed(123)
#env.seed(123)
nb_actions = env.action_space.n
print(nb_actions)
print(env.observation_space.shape)
print((1,) + env.observation_space.shape)

indsize = env.get_indicator_size(setting["indicator"])
input_shape = (WINDOW_LENGTH,) + (INPUT_SHAPE, indsize)
# Next, we build a very simple model.
model = Sequential()
#model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(nb_actions, input_shape=(144,16)))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if args.mode == "train-load":
    dqn.load_weights(args.weights)

if args.mode == "train-load" or args.mode == "train":
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100)]
    callbacks += [FileLogger(log_filename, interval=100)]
    #callbacks += [TensorBoard(log_dir=dirname, histogram_freq=1, write_grads=True)]
    dqn.fit(env, callbacks=callbacks, nb_steps=250000, visualize=True, verbose=2)

# After training is done, we save the final weights.

# After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=1, visualize=True)

if args.mode == "test":
    dqn.load_weights(args.weights)

# Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=1, visualize=True)
    print(json.dumps(env.report(), indent=2))
