import gym
from gym import spaces
import numpy as np
from backtest import Backtest
import config

from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)


BUY, SELL, HOLD = range(3)

class BacktestEnv(gym.Env, Backtest):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.setting = config.get_setting()
        indsize = self.get_indicator_size(self.setting["indicator"])
        maxnum = np.array([100.0] * indsize)
        #self.observation_space = spaces.Box(-maxnum, maxnum)
        self.observation_space = spaces.Box(-maxnum, maxnum)
        self.action_space = spaces.Discrete(3)
        self.init_candle(candleSize=10, deltaDays=30, random=False)
        super(BacktestEnv, self).__init__()

    def _reset(self):
        status = self.resetbacktest()
        self.update_indicators()
        return status

    def _step(self, action):
        idx = self.index + 1
        price = self.candles[self.index, self.col["close"]]
        date = self.candles[self.index, self.col["start"]]
        is_action = self.stepAction(action, price, date, "rl")
        reward = self.get_reward(is_action)
        status = self.get_status()
        index_over = idx >= self.size
        #index_over = idx >= self.size or self.alpha[self.index] < -5. or (self.index > 10 and self.index / 100 > self.alpha[self.index] + 10)
        #index_over = idx >= self.size or self.alpha[self.index] < -2. or (idx > 10 and self.max_alpha / 2 > self.alpha[self.index])
        actionname = "HOLD"
        if is_action:
            actionname = self.get_actionname(action)
        print("index:%d, action:%s, alpha_diff:%+.2f, reward:%+.2f, alpha:%+.2f" % (self.index, actionname, self.alpha_diff[self.index], reward, self.alpha[self.index]))
        self.index += 1
        return status, reward, index_over, {}

    def _get_reward(self):
        return self.get_reward()

    def _render(self, mode='human', close=False):
        pass
