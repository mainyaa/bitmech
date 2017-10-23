#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import poloniex
import time
import datetime
import json
import math
import argparse
import random
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
from scipy.signal import lfilter
import talib
import tqdm

import indicators
import indicatorsnumpy
import config
from finance import test_risk_metrics, test_risk_adjusted_metrics

from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)

maxnum = np.finfo(np.float32).max
MA_SMA, MA_EMA, MA_WMA, MA_DEMA, MA_TEMA, MA_TRIMA, MA_KAMA, MA_MAMA, MA_T3 = range(9)

# rename column: from -> to
renamemigrate = {
        "RSI": {
                "interval": "n",
                },
        "STOCHRSI": {
                "interval": "n",
                },
        }
renametalib = {
        "DEMA": {
                "timeperiod": "n_slow",
                },
        "MACD": {
                "fastperiod": "n_fast",
                "slowperiod": "n_slow",
                "signalperiod": "signal",
                },
        "PPO": {
                "fastperiod": "n_fast",
                "slowperiod": "n_slow",
                },
        "RSI": {
                "timeperiod": "interval",
                },
        "STOCHRSI": {
                "timeperiod": "interval",
                "fastk_period": "interval",
                "fastd_period": "interval",
                },
        "CCI": {
                "timeperiod": "history",
                },
        }
src = dict(
    index = 'start',
    op = 'open',
    hi = 'high',
    lo = 'low',
    cl = 'close',
    aop = None,
    ahi = None,
    alo = None,
    acl = None,
    vo = 'volume',
    di = None,
)
BUY, SELL, HOLD = range(3)
OPEN, HIGH, LOW, CLOSE, VWP, VOLUME, DATE = range(7)
dict_merge = lambda a,b: a.update(b) or a
def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "-" + subkey, subvalue
            else:
                yield key, value
    return dict(items())

def compressing_flatten_dict(IND):
    conf = {}
    for key2 in IND.keys():
        if key2.find("-") != -1:
            k = key2.split('-')
            if k[0] not in conf:
                conf[k[0]] = {}
            conf[k[0]][k[1]] = IND[key2]
            if k[1] == "persistence" or k[1] == "history":
                conf[k[0]][k[1]] = int(round(IND[key2]))
        elif key2 == "longWeight":
            conf["long"] = IND["short"] * IND[key2]
        else:
            conf[key2] = IND[key2]
    return conf

class Backtest(object):
    def __init__(self):
        self.setting = config.get_setting()
        self.indicator = self.setting["indicator"]
        self.indsize = self.get_indicator_size(self.indicator)
        self.params = {}
        for k in self.indicator:
            self.params[k] = self.setting[k]
        self.result = []
        self.trade_result = []
        self.talib = False


    def datetime_to_epoch(self, d):
        if type(d) is np.datetime64:
            a = np.array([d], dtype='M8[us]')
            return a.astype('uint64') / 1e6
        else:
            return time.mktime(d.timetuple())

    def getRandomDateRange(self, Limits, deltaDays, testDays=0):
        epochToDatetime = lambda D: datetime.datetime.utcfromtimestamp(D)
        FLms = self.datetime_to_epoch(Limits['start'])
        TLms = self.datetime_to_epoch(Limits['end'])
        deltams=deltaDays * 24 * 60 * 60
        testms=testDays * 24 * 60 * 60

        Starting= random.randint(FLms,TLms-deltams-testms)
        DateRange = {
            "start": epochToDatetime(Starting),
            "end": epochToDatetime(Starting+deltams)
        }

        return DateRange

    def init_candle(self ,candleSize=10, deltaDays=-1, random=True):
        nsec = int(candleSize*6*10e9) # minute end nanosecond
        df = self.load_bitflyer()
        #df = self.load_poloniex('USDT_BTC')
        dates = {
                "start": None,
                "end": None,
                }
        # random date if deltaDays != -1
        if deltaDays != -1:
            dates["start"] = pd.to_datetime(df.ix[0, ["start"]])
            dates["start"] = datetime.datetime.combine(dates["start"].dt.date[0], dates["start"].dt.time[0])
            dates["end"] = pd.to_datetime(df.ix[-1, ["start"]])
            dates["end"] = datetime.datetime.combine(dates["end"].dt.date[0], dates["end"].dt.time[0])
            if random:
                dates = self.getRandomDateRange(dates, deltaDays)
            else:
                dates["start"] = dates["end"] - datetime.timedelta(days=deltaDays)
        df = Backtest.resample_candle_range(df, "%sN" % nsec, dates["start"], dates["end"])
        self.set_df(df)

    def set_df(self, df):
        self.df = df
        self.size = self.df.shape[0]
        self.col = {x: self.df.columns.tolist().index(x) for x in ["open", "high", "low", "close", "vwp", "volume", "start"]}
        self.candles = self.df.values
        if "candleSize" in self.setting["CCI"]:
            self.CCIcandles = self.df.resample(int(self.setting["CCI"]["candleSize"])+"min").ohlc().values
        else:
            self.CCIcandles = self.candles

    def get_indicator_size(self, indicator):
        indsize = len(indicator) + 1
        if "BBANDS" in indicator:
            indsize += 1
        if "MACD" in indicator or "PPO" in indicator or "KELCH" in indicator:
            indsize += 2
        return indsize

    def reset_portfolio(self):
        price = self.candles[0, self.col["close"]]
        portfolio = {
            "asset": 1,
            "currency": 100,
            "price": price,
            "balance": False,
            }
        portfolio["balance"] = portfolio["currency"] + price * portfolio["asset"]
        return portfolio

    def setparams(self, params):
        key = self.strategy.split("_")[0]
        self.params[key] = params

    def resetbacktest(self):
        self.portfolio = self.reset_portfolio()
        self.index = 0
        self.last_action_index = self.index
        self.alpha = np.zeros((self.size,), dtype=np.float64)
        self.start = self.portfolio.copy()
        self.balance = np.zeros((self.size,), dtype=np.float64)
        self.relative_profit = np.zeros((self.size,), dtype=np.float64)
        self.relative_profit_diff = np.zeros((self.size,), dtype=np.float64)
        self.alpha = np.zeros((self.size,), dtype=np.float64)
        self.alpha_diff = np.zeros((self.size,), dtype=np.float64)
        self.last_action = None
        self.status = np.zeros((self.size, self.indsize))
        self.status_result = np.zeros((self.size, self.indsize))
        self.trades = 0
        for name in self.indicator:
            self.init_trend(name, "hold")
        return self.status

    def stepActions(self, action, price, date, indicator):
        historySize = 10
        if type(action) is int:
            action = [action]
        for i, act in enumerate(action):
            if "hisotrySize" in self.params[indicator]:
                historySize = int(self.params[indicator]["hisotrySize"])
            if historySize >= i:
                continue
            self.stepAction(action[i], price[i], date[i], indicator)
        return self.trade_result

    def stepAction(self, action, price, date, indicator):
        if action == BUY:
            if self.portfolio["currency"] == 0:
                return False
            self.portfolio = self.exec_long(price, self.portfolio, date)
            trade = self.handle_trade("buy", price, self.portfolio, date, indicator)
            self.trades += 1
            self.trade_result.append(trade)
            return True
        if action == SELL:
            if self.portfolio["asset"] == 0:
                return False
            self.portfolio = self.exec_short(price, self.portfolio, date)
            trade = self.handle_trade("sell", price, self.portfolio, date, indicator)
            self.trades += 1
            self.trade_result.append(trade)
            return True
        return False

    def get_actionname(self, action):
        return [" BUY", "SELL", "HOLD"][action]

    def get_reward(self, is_action=True):
        price = self.candles[self.index, self.col["close"]]
        market = (100 * price / self.start["price"]) - 100
        balance = self.portfolio["currency"] + price * self.portfolio["asset"]
        alpha = balance / self.start["balance"] * 100 - 100 - market
        self.alpha[self.index] = alpha
        diff = alpha - self.alpha[self.last_action_index]
        self.alpha_diff[self.index] = diff
        self.status[self.index, -1] = diff
        if not is_action:
            return 0.
        self.last_action_index = self.index
        if diff > 0:
            return diff * 2
        return diff

    def migrate_dict(self, key, params):
        newparams = params.copy()
        if key not in renamemigrate:
            return newparams
        for old, new in renamemigrate[key].items():
            if old not in newparams:
                continue
            newparams[new] = newparams[old]
            del newparams[old]
        return newparams

    def talib_dict(self, params):
        # dict key rename
        newparams = {}
        for k in renametalib.keys():
            if k not in params:
                continue
            newparams[k] = {}
            for new, old in renametalib[k].items():
                newparams[k][new] = params[k][old]
        # add matype
        if "PPO" in params:
            newparams["PPO"]["matype"] = MA_EMA
        if "STOCHRSI" in params:
            newparams["STOCHRSI"]["fastd_matype"] = MA_EMA

        return newparams

    def load_json(self, filename):
        f = open(filename, "r")
        result = json.loads(f.read())
        f.close
        return result

    def exec_long(self, price, portfolio, date):
        currency = portfolio["currency"]
        if currency == 0:
            return portfolio
        portfolio["asset"] += self.extract_fee(currency / price)
        portfolio["currency"] = 0
        portfolio["balance"] = portfolio["currency"] + price * portfolio["asset"]
        return portfolio

    def exec_short(self, price, portfolio, date):
        asset = portfolio["asset"]
        if asset == 0:
            return portfolio
        portfolio["currency"] += self.extract_fee(asset * price)
        portfolio["asset"] = 0
        portfolio["balance"] = portfolio["currency"] + price * portfolio["asset"]
        return portfolio

    def handle_trade(self, position, price, portfolio, date, indicator):
        trade = {
              "action": position,
              "price": price,
              "portfolio": portfolio.copy(),
              "balance": portfolio["balance"],
              "relative_profit": 100 * portfolio["balance"] / self.start["balance"] - 100,
              "date": date,
              "indicator": indicator,
              }
        if len(self.trade_result) == 0:
            prev = self.start
            trade["profitAndLoss"] = trade["balance"] - prev["balance"]
            trade["profit"] = (100 * trade["balance"] / prev["balance"]) - 100
            trade["market"] = (100 * price / prev["price"]) - 100
        else:
            prev = self.trade_result[-1]
            trade["profitAndLoss"] = trade["balance"] - prev["balance"]
            trade["profit"] = (100 * trade["balance"] / prev["balance"]) - 100
            trade["market"] = (100 * price / prev["price"]) - 100
        return trade

    def extract_fee(self, amount):
        fee_maker = 0.15
        slippage = 0.05
        fee = 1-(fee_maker+slippage)/100
        amount *= 1e8
        amount *= fee
        amount = math.floor(amount)
        amount /= 1e8
        return amount

    def init_trend(self, name, direction):
        if not hasattr(self, "trend"):
            self.trend  = {}
        self.trend[name] = {
                "duration": 0,
                "persisted": False,
                "direction": direction,
                "adviced" : False,
                }
        return self.trend[name]

    def get_status(self):
        self.status[self.index, :] = self.status_result[self.index, :]
        return self.status

    def update_indicators(self):
        idx = self.size + 1
        candle = self.candles[:, self.col["close"]]
        for i, key in enumerate(self.indicator):
            params = self.params[key]
            """
            if "candleSize" in params:
                candleSize = int(params["candleSize"])
            if "hisotrySize" in params:
                historySize = int(params["hisotrySize"])
            if self.index % candleSize != 0:
                pass
                #continue
            if candleSize * historySize >= self.index:
                continue
            """
            if self.talib:
                key = key.split("_")[0]
                if not hasattr(talib, key):
                    raise Exception(key + " is not implimented.")
                d = self.talib_dict(self.params)
                newargs = d[key]
                # Dynamic calling
                fn = getattr(talib, key)
                if key == "CCI":
                    high = self.CCIcandles[:idx, self.col["high"]]
                    low = self.CCIcandles[:idx, self.col["low"]]
                    result = fn(high, low, candle, **newargs)
                else:
                    result = fn(candle, **newargs)
                result = np.nan_to_num(result)
                ressize = 1
                # not 1-D array
                if len(result.shape) > 1:
                    ressize = result.shape[1]
                self.status_result[:idx, i:i+ressize] = result
            else:
                # Dynamic calling
                key = key.split("_")[0]
                if not hasattr(indicatorsnumpy, key):
                    raise Exception(indicatorsnumpy + " is not implimented.")
                fn = getattr(indicatorsnumpy, key)
                #fn = getattr(self, key)
                args = self.get_actionparams(params)
                args["df"] = candle
                result = fn(**args)
                self.status_result[:idx, i] = result

    def actionIndicators(self, key):
        key = key.split("_")[0]
        #self.action(self.df, key, self.params[key], self.stepAction) # actionGeneral
        self.actionnumpy(self.candles[:, self.col["close"]], key, self.params[key], self.stepAction) # actionGeneral
        #self.actiontalib(self.candles[:, self.col["close"]], key, self.params[key], self.stepAction) # actionGeneral

    def action(self, df, name, params, callback):
        # Dynamic calling
        fn = getattr(indicators, name)
        up = "high"
        down = "low"
        if "up" in params["thresholds"]:
            up = "up"
            down = "down"
        if "persistence" not in params["thresholds"]:
            params["thresholds"]["persistence"] = 1
        args = self.get_actionparams(params)
        args["df"] = df
        result = fn(**args)
        return self.actionGeneral(
                name=name,
                result=result,
                up=params["thresholds"][up],
                down=params["thresholds"][down],
                persistence=params["thresholds"]["persistence"],
                callback=callback,
                )
    def actionnumpy(self, price, name, params, callback):
        # Dynamic calling
        fn = getattr(indicatorsnumpy, name)
        up = "high"
        down = "low"
        if "up" in params["thresholds"]:
            up = "up"
            down = "down"
        if "persistence" not in params["thresholds"]:
            params["thresholds"]["persistence"] = 1
        args = self.get_actionparams(params)
        args["df"] = price
        result = fn(**args)
        return self.actionGeneral(
                name=name,
                result=result,
                up=params["thresholds"][up],
                down=params["thresholds"][down],
                persistence=params["thresholds"]["persistence"],
                callback=callback,
                )
    def actiontalib(self, price, name, params, callback):
        # Dynamic calling
        fn = getattr(talib, name)
        #print(fn.__doc__)
        up = "high"
        down = "low"
        if "up" in params["thresholds"]:
            up = "up"
            down = "down"
        if "persistence" not in params["thresholds"]:
            params["thresholds"]["persistence"] = 1
        d = {}
        d[name] = params
        args = self.talib_dict(d[name])
        args["real"] = np.array(price, dtype=np.float64)
        result = fn(**args)
        return self.actionGeneral(
                name=name,
                result=result,
                up=params["thresholds"][up],
                down=params["thresholds"][down],
                persistence=params["thresholds"]["persistence"],
                callback=callback,
                )
    def get_actionparams(self, params):
        p = params.copy()
        del p["candleSize"]
        del p["historySize"]
        del p["thresholds"]
        return p

    def actionGeneral(self, name, result, up, down, persistence, callback):
        if isinstance(result, tuple):
            result = result[0]
        actions = np.zeros((result.shape))
        for i, value in enumerate(result):
            if value is None:
                value = 0
            trend = self.trend[name]
            act = HOLD
            if value > up:
                if trend["direction"] != up:
                    if trend["direction"] == down and trend["adviced"]:
                        callback(BUY)
                    trend = self.init_trend(name, up)
                trend["duration"] += 1
                if trend["duration"] >= persistence:
                    trend["persisted"] = True
                if trend["persisted"] and not trend["adviced"]:
                    trend["adviced"] = True
                    act = BUY
                    callback(act)
            elif value < down:
                if trend["direction"] != down:
                    if trend["direction"] == up and trend["adviced"]:
                        callback(SELL)
                    trend = self.init_trend(name, down)
                trend["duration"] += 1
                if trend["duration"] >= persistence:
                    trend["persisted"] = True
                if trend["persisted"] and not trend["adviced"]:
                    trend["adviced"] = True
                    act = SELL
                    callback(act)
            self.trend[name] = trend
            actions[i] = act
            #self.balance[i] = self.portfolio["currency"] + price * self.portfolio["asset"]
            #self.relative_profit[i] = self.balance[i] / self.balance[0] * 100 - 100
        return actions, result

    @staticmethod
    def load_csv():
        files = ["data/coincheckJPY.csv"]
        return files

    @staticmethod
    def detect_outliers(trade):
        return trade[trade.price.diff() > -100000]

    @staticmethod
    def resample_candle(trade, exchange="bitcoincharts", freq="1min"):
        amount_col = "amount"
        date_col = "start"
        price_col = "price"
        if exchange == "coincheck":
            price_col = "rate"
            date_col = "created_at"
        if exchange == "bitflyer":
            amount_col = "size"
            date_col = "exec_date"
        trade = Backtest.detect_outliers(trade)

        # resample
        ohlc = trade[price_col].resample(freq).ohlc().fillna(method="backfill")
        start = pd.DataFrame(ohlc.index)
        start.index = start[date_col]
        volume = trade[amount_col].resample(freq).sum().fillna(method="backfill")
        trade_count = trade[amount_col].resample(freq).count().fillna(0)
        vwp = trade[price_col].multiply(trade[amount_col]).resample(freq).sum().divide(volume).fillna(method="backfill")

        candle = pd.concat([start, ohlc, vwp, volume, trade_count], axis=1, join='inner')
        candle.columns = ["start", "open", "high", "low", "close", "vwp", "volume", "trades"]
        candle = candle.fillna(method="backfill")
        candle.index = candle["start"]
        for col in "open,high,low,close,vwp,volume".split(","):
            candle[col] = candle[col].astype(np.float64)
        return candle

    @staticmethod
    def is_update_coincheck():
        if not os.path.exists("data/coincheckJPY.csv"):
            return True
        if not os.path.exists("data/coincheckJPY_1min.csv"):
            return True
        return os.path.getmtime("data/coincheckJPY.csv") > os.path.getmtime("data/coincheckJPY_1min.csv")
    @staticmethod
    def is_update_bitflyer():
        if not os.path.exists("data/bitflyerBTC_JPY_trades.csv"):
            return True
        if not os.path.exists("data/bitflyerBTC_JPY_1min.csv"):
            return True
        return os.path.getmtime("data/bitflyerBTC_JPY_trades.csv") > os.path.getmtime("data/bitflyerBTC_JPY_1min.csv")
    @staticmethod
    def is_update_poloniex():
        if not os.path.exists("data/poloniexUSDT_BTC_5min.csv"):
            return True
        return False

    @staticmethod
    def load_coincheck_trade():
        files = Backtest.load_csv()
        trade = pd.read_csv(files[-1], header=None)
        trade.columns = ["start", "price", "amount"]
        trade["start"] = pd.to_datetime(trade["start"], unit='s')
        trade.index = trade["start"]
        return trade

    @staticmethod
    def load_coincheck():
        # get last modified time
        if Backtest.is_update_coincheck():
            trade = Backtest.load_coincheck_trade()
            candle = Backtest.resample_candle(trade, exchange="coincheck", freq="1min")
            candle.to_csv("data/coincheckJPY_1min.csv", index=None)
        else:
            candle = pd.read_csv("data/coincheckJPY_1min.csv", parse_dates=[0])
            candle.index = candle["start"]
        return candle

    @staticmethod
    def load_bitflyer():
        # get last modified time
        if Backtest.is_update_bitflyer():
            trade = Backtest.load_bitflyer_trade()
            candle = Backtest.resample_candle(trade, exchange="bitflyer", freq="1min")
            candle.index = candle["start"]
            candle.to_csv("data/bitflyerBTC_JPY_1min.csv", index=None)
            print(candle)
        else:
            candle = pd.read_csv("data/bitflyerBTC_JPY_1min.csv", parse_dates=[0])
            candle.index = candle["start"]
        return candle

    @staticmethod
    def select_daterange(candle, start=None, end=None):
        if start is None and end is None:
            return candle
        if start is None:
            start = datetime.datetime.now() - datetime.timedelta(days=90)
        if end is None:
            end = datetime.datetime.now()
        result = candle[(candle.start.dt.to_pydatetime() >= start) & (candle.start.dt.to_pydatetime() <= end)]
        return result

    @staticmethod
    def resample_candle_range(candle, freq="10min", start=None, end=None):
        candle = Backtest.resample_ohlc(candle, freq)
        candle = Backtest.select_daterange(candle, start, end)
        return candle

    @staticmethod
    def resample_ohlc(candle, freq="10min"):
        columns = ["start", "open", "high", "low", "close", "vwp", "volume", "trades"]
        o = candle["open"].resample(freq).first().fillna(method="backfill")
        start = pd.DataFrame(o.index)
        start.index = start["start"]
        h = candle["high"].resample(freq).max().fillna(method="backfill")
        l = candle["low"].resample(freq).min().fillna(method="backfill")
        c = candle["close"].resample(freq).last().fillna(method="backfill")
        v = candle["volume"].resample(freq).sum().fillna(method="backfill")
        if "trades" in candle:
            t = candle["trades"].resample(freq).sum().fillna(0)
        else:
            t = pd.DataFrame([])
            columns.remove("trades")
        if "vwp" in candle:
            vwp = candle["vwp"].resample(freq).mean().fillna(method="backfill")
        else:
            vwp = pd.DataFrame([])
            columns.remove("vwp")
        join_list = [ x for x in [start, o, h, l, c, vwp, v, t] if len(x) > 0]
        result = pd.concat(join_list, axis=1, join='inner')
        result.columns = columns
        return result

    @staticmethod
    def load_bitflyer_trade():
        if os.path.exists("data/bitflyerBTC_JPY_trades.csv"):
            exe = pd.read_csv("data/bitflyerBTC_JPY_trades.csv", parse_dates=[1])
            exe.columns = ["buy_child_order_acceptance_id", "exec_date", "id", "price", "sell_child_order_acceptance_id", "side", "size"]
            print(exe)
            exe.index = exe["exec_date"]
        else:
            exe = Backtest.load_bitflyer_server()
        return exe

    @staticmethod
    def load_coincehck_server():
        import coincheck.coincheck
        import config
        setting = config.get_setting()
        api = coincheck.coincheck.CoinCheck(accessKey=setting["coincheck"]["api_key"], secretKey=setting["coincheck"]["api_secret"])
        trade = api.trade.all()
        trade = pd.DataFrame.from_dict(trade)
        trade["created_at"] = pd.to_datetime(trade["created_at"])
        trade.index = trade["created_at"]

        # get bitflyer for target date(today - 90)
        target_date = datetime.datetime.today() - datetime.timedelta(days=90)
        end_date = trade.iloc[-1,1]
        print(target_date, end_date)
        print(trade)
        print(trade.shape)
        offset = trade.shape[0] - 1
        print(offset)
        """
        while target_date < end_date:
            end_date = trade.iloc[-1,1]
            e1 = api.trade.all({"offset":offset})
            e1 = pd.DataFrame.from_dict(e1)
            print(e1)
            e1["created_at"] = pd.to_datetime(e1["created_at"])
            e1.index = e1["created_at"]
            trade = trade.append(e1)
            trade.drop_duplicates(["id"], inplace=True)
            offset = trade.shape[0] - 1
            print(offset)
            time.sleep(1)
        """
        print(trade)
        trade.to_csv("data/coincheckBTC_JPY_trade.csv", index=None)
        return trade

    @staticmethod
    def load_zaif_server():
        from zaifapi import ZaifPublicApi
        zaif = ZaifPublicApi()
        trade = zaif.trades(currency_pair='btc_jpy',
                   limit=200000)
        return trade

    @staticmethod
    def load_bitflyer_server():
        import config
        import pybitflyer
        setting = config.get_setting()
        api = pybitflyer.API(api_key=setting["bitflyer"]["api_key"], api_secret=setting["bitflyer"]["api_secret"])
        executions = api.executions(product_code="BTC_JPY", count=500)
        executions = pd.DataFrame.from_dict(executions)
        executions["exec_date"] = pd.to_datetime(executions["exec_date"])
        executions.index = executions["exec_date"]
        print(executions)

        # get bitflyer for target date(today - 90)
        target_date = datetime.datetime.today() - datetime.timedelta(days=365)
        end_date = executions.iloc[-1,1]
        while target_date < end_date:
            end_date = executions.iloc[-1,1]
            print(end_date)
            before = executions.iloc[-1,2]
            e1 = api.executions(product_code="BTC_JPY", count=500, before=before)
            e1 = pd.DataFrame.from_dict(e1)
            e1["exec_date"] = pd.to_datetime(e1["exec_date"])
            e1.index = e1["exec_date"]
            executions = executions.append(e1)
            time.sleep(1)
        executions.to_csv("data/bitflyerBTC_JPY_executions.csv", index=None)
        return executions


    @staticmethod
    def load_bitflyer_server_365():
        import config
        import pybitflyer
        setting = config.get_setting()
        api = pybitflyer.API(api_key=setting["bitflyer"]["api_key"], api_secret=setting["bitflyer"]["api_secret"])
        i = 12354
        executions = pd.read_csv("data/bitflyerBTC_JPY_executions_%d.csv" % i)

        # get bitflyer for target date(today - 90)
        target_date = datetime.datetime.today() - datetime.timedelta(days=365)

        end_date = datetime.datetime.strptime(executions.iloc[-1,1], '%Y-%m-%d %H:%M:%S.%f')
        before = executions.iloc[-1,2]
        while target_date < end_date:
            print(end_date)
            e1 = api.executions(product_code="BTC_JPY", count=500, before=before)
            e1 = pd.DataFrame.from_dict(e1)
            e1["exec_date"] = pd.to_datetime(e1["exec_date"])
            e1.index = e1["exec_date"]
            end_date = e1.iloc[-1,1]
            before = e1.iloc[-1,2]
            time.sleep(1)
            e1.to_csv("data/bitflyerBTC_JPY_executions_%s.csv"%i, index=None, header=None)
            i += 1
        executions.to_csv("data/bitflyerBTC_JPY_executions.csv", index=None)
        return executions


    @staticmethod
    def load_poloniex(pair='USDT_BTC', period=None, start=None, end=None):
        if Backtest.is_update_poloniex():
            candle = Backtest.load_poloniex_trade(pair='USDT_BTC', period=None, start=None, end=None)
            candle.to_csv("data/poloniexUSDT_BTC_5min.csv", index=None)
        else:
            candle = pd.read_csv("data/poloniexUSDT_BTC_5min.csv", parse_dates=[1])
            candle.index = candle["start"]
        for col in "open,high,low,close,vwp,volume".split(","):
            candle[col] = candle[col].astype(np.float64)
        return candle

    @staticmethod
    def load_poloniex_trade(pair='USDT_BTC', period=None, start=None, end=None):
        polo = poloniex.Poloniex()
        polo.timeout = 2
        # "period" (candlestick period in seconds; valid values are 300, 900, 1800, 7200, 14400, and 86400)
        if period is None:
            period = 60*5
        if start is None:
            start = time.time()-60*60*24*365 # 100 days
        chart = polo.returnChartData(pair, period=period, start=start, end=time.time())
        chart = pd.DataFrame.from_dict(chart).astype(np.float64)
        chart["date"] = pd.to_datetime(chart["date"], unit='s')
        chart = chart.rename(columns={"date": "start", "weightedAverage": "vwp"})
        chart.index = chart["start"]
        return chart

    def report(self, freq="7d"):
        dates = {
                "start": self.candles[0, self.col["start"]],
                "end": self.candles[-1, self.col["start"]],
                }
        timespan = dates["start"] - dates["end"]
        # the portfolio's balance is measured in {currency}
        startPrice = self.candles[0, self.col["close"]]
        endPrice = self.candles[-1, self.col["close"]]
        start = self.reset_portfolio()
        self.portfolio["balance"] = self.portfolio["currency"] + endPrice * self.portfolio["asset"]
        relative_profit = (100 * self.portfolio["balance"] / start["balance"]) - 100
        profit = self.portfolio["balance"] - start["balance"]

        report = {
                "startTime": dates["start"].strftime('%Y-%m-%d %H:%M:%S'),
                "endTime": dates["end"].strftime('%Y-%m-%d %H:%M:%S'),
                "timespan": str(timespan),
                "market": endPrice * 100 / startPrice - 100,

                "profit": profit,
                "relative_profit": relative_profit,

                #"yearlyProfit": round(profit / timespan.year()),
                #"relativeYearlyProfit": round(relative_profit / timespan.year()),
                "trades": self.trades,
        }
        profitdf = self.df.copy()
        profitdf["balance"] = self.balance
        profitdf["relative_profit"] = self.relative_profit
        balance = profitdf["balance"].resample(freq).last().fillna(0).values
        close = profitdf["close"].resample(freq).last().fillna(0).values
        relative_profit = profitdf["relative_profit"].resample(freq).last().diff(1).fillna(0).values

        report["alpha"] = report["relative_profit"] - report["market"]
        risk = test_risk_metrics(relative_profit, close)
        risk2 = test_risk_adjusted_metrics(relative_profit, close)
        report.update(risk)
        report.update(risk2)
        report["alpha2"] = report["relative_profit"] - report["beta"] * report["market"]
        report["config"] = self.params[self.strategy].copy()
        self.result.append(pd.DataFrame.from_dict([report]))
        return report

    def run_from_json(self, inputfile):
        filelist = glob.glob(inputfile)
        for f in tqdm.tqdm(filelist):
            jsonconfig = self.load_json(f)["gekkoConfig"]
            self.strategy = f.split("/")[-1].split("_")[3]
            if self.strategy not in self.indicator:
                continue
            backtestReturn = float(f.split("_")[-2])
            self.setting[self.strategy] = jsonconfig[self.strategy]
            self.params[self.strategy] = jsonconfig[self.strategy]
            if "hith" in self.params[self.strategy]["thresholds"]:
                self.params[self.strategy]["thresholds"]["high"] = self.params[self.strategy]["thresholds"]["hith"]
            if "candleSize" in self.params[self.strategy]:
                self.init_candle(self.params[self.strategy]["candleSize"])
            else:
                self.init_candle()
            self.resetbacktest()
            self.actionIndicators(self.strategy)
            report = self.report()
            report["file"] = f
            report["strategy"] = self.strategy
            report["backtestReturn"] = backtestReturn
        r = pd.concat(self.result)
        print("="*50)
        print(r)
        filename = './data/result.from_json.{}.{}.csv'.format(self.strategy, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        r.to_csv(filename, index=None)

    def run_from_text(self, inputfile):
        filetext = open(inputfile).readlines()
        filetext = np.unique(filetext)
        self.strategy = inputfile.split("/")[-1].split(".")[0]
        params = [self.migrate_dict(self.strategy, eval(f)) for f in filetext]
        for param in tqdm.tqdm(params):
            self.run_single(param, deltaDays=365, random=False)
        params = [flatten_dict(eval(f)) for f in filetext]
        max_param = pd.DataFrame.from_dict(params).max()
        min_param = pd.DataFrame.from_dict(params).min()
        print(max_param, min_param)
        r = pd.concat(self.result)
        print("="*50)
        print(r)
        filename = './data/result.from_text.{}.{}.csv'.format(self.strategy, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        r.to_csv(filename, index=None)

    def run_single(self, params, deltaDays=-1, random=True):
        p = compressing_flatten_dict(params)
        if "hith" in p["thresholds"]:
            p["thresholds"]["high"] = p["thresholds"]["hith"]
        arg = {
                "candleSize": 10,
                "deltaDays": -1,
                }
        if "candleSize" in params:
            arg["candleSize"] = params["candleSize"]
        if deltaDays != -1:
            arg["deltaDays"] = deltaDays
        elif "deltaDays" in self.setting["hyperopt"]:
            arg["deltaDays"] = self.setting["hyperopt"]["deltaDays"]
        arg["random"] = random
        self.init_candle(**arg)
        self.resetbacktest()
        self.setparams(p)
        self.actionIndicators(self.strategy)
        report = self.report("7d")
        if len(self.trade_result) == 0:
            return -report["alpha"]
        return -report["alpha"]

    def run(self):
        for i, key in enumerate(self.indicator):
            self.strategy = key
            self.run_single(self.params[key], deltaDays=365, random=False)
        r = pd.concat(self.result)
        r = r.reset_index()
        print(r)
        print(json.dumps(r.to_dict(), indent=2))

    def run_boost(self):
        setting = config.get_setting()
        num_rounds = setting["hyperopt"]["num_rounds"]
        deltaDays = setting["hyperopt"]["deltaDays"]
        for i in tqdm.tqdm(range(num_rounds)):
            self.strategy = "RSI"
            self.run_single(self.params[self.strategy], deltaDays=deltaDays, random=True)
        r = pd.concat(self.result)
        r = r.reset_index()
        print(r)
        print(json.dumps(r.to_dict(), indent=2))
        filename = './data/result.config-boost.{}.{}.csv'.format(self.strategy, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        r.to_csv(filename, index=None)

def run_optimize(params):
    global optimize_report
    result = []
    setting = config.get_setting()
    num_rounds = setting["hyperopt"]["num_rounds"]
    skip = 0
    for i in tqdm.tqdm(range(num_rounds)):
        backtest = Backtest()
        backtest.strategy = "RSI"
        res = backtest.run_single(params, deltaDays=-1, random=True)
        result.append(res)
        if skip > 1 and skip > num_rounds/10 and i < num_rounds/5:
            print("%f: num_rounds/10:" % np.mean(result), params)
            return np.mean(result)
        if backtest.trades < 2:
            skip += 1
            continue
        if len(backtest.result) > 1:
            if backtest.result["alpha"][-1] < 0 or backtest.result["relative_profit"][-1] < 50 or backtest.result["treynor_ratio"][-1] < 1:
                skip += 1
                continue
        optimize_report.append(pd.concat(backtest.result))
    r = np.mean(result)
    print(r, params)
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, choices=['config', 'config-boost', 'json', 'text', 'optimize'])
    parser.add_argument('--input', type=str, default="../gekkoJaponicus/*/*_config.json")
    args = parser.parse_args()
    if args.run == "config":
        backtest = Backtest()
        backtest.run()
    if args.run == "config-boost":
        backtest = Backtest()
        backtest.run_boost()
    elif args.run == "json":
        backtest = Backtest()
        backtest.run_from_json(args.input)
    elif args.run == "text":
        backtest = Backtest()
        backtest.run_from_text(args.input)
    elif args.run == "optimize":
        optimize_report = []
        backtest = Backtest()
        backtest.strategy = "RSI"
        #trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp2')
        trials = Trials()
        best = fmin(fn=run_optimize,
            space=backtest.setting["hyperopt"][backtest.strategy],
            algo=tpe.suggest,
            trials=trials,
            max_evals=backtest.setting["hyperopt"]["num_iter"])
        result = []
        num_rounds = backtest.setting["hyperopt"]["num_rounds"]
        backtest.run_single(best, deltaDays=364)
        report = backtest.report()
        trade_result = pd.DataFrame.from_dict(backtest.trade_result)
        print("="*50)
        print("original report")
        print("="*50)
        print(best)
        print("="*50)
        print(trade_result)
        print("="*50)
        print(json.dumps(report, indent=2))
        print(optimize_report)
        r = pd.concat(optimize_report)
        print(r)
        filename = './data/result.optimize.{}.{}.csv'.format(backtest.strategy, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        r.to_csv(filename, index=None)
    else:
        parser.print_help()
