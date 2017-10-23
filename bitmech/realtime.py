#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np
import argparse
import tqdm
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')


import config
import exchange
from backtest import Backtest
from realtimeexchange import RealtimeExchange
from finance import test_risk_metrics, test_risk_adjusted_metrics

from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)
dict_merge = lambda a,b: a.update(b) or a
import inspect

BUY, SELL, HOLD = range(3)

def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr

class Realtime(object):
    def __init__(self):
        self.setting = config.get_setting()
        self.indicators = self.setting["indicator"]
        self.params = {}
        for k in self.indicators:
            self.params[k] = self.setting[k]
        self.result = []
        self.trade_result = []
        self.talib = False
        self.exchangename = self.setting["realtime"]["exchange"]
        self.pair = self.setting["realtime"]["pair"]
        self.currency = self.pair.split("_")[0]
        self.asset = self.pair.split("_")[1]
        self.strategy = self.setting["realtime"]["strategy"]
        print(self.setting["realtime"])
        print(self.params[self.strategy])
        self.exchange = getattr(exchange, self.exchangename)(self.setting)
        self.realtimeexchange = RealtimeExchange()
        self.backtest = Backtest()
        self.fee = self.exchange.get_fee()
        self.balance = self.exchange.getbalance()
        self.executions = None
        self.order_id = None
        self.is_dryrun = False

    def spread(self):
        executions = self.exchange.gettrade(datetime.datetime.now(), 50)
        print(executions)
        ticker = self.exchange.ticker()
        print(ticker)
        board = self.exchange.board()
        mid_price = board["mid_price"]
        bids = pd.DataFrame.from_dict(board["bids"])
        asks = pd.DataFrame.from_dict(board["asks"])
        bids["type"] = "bids"
        asks["type"] = "asks"
        askbid = pd.concat([bids, asks])
        askbid.index = askbid["price"]
        askbid = askbid.sort_values("price")
        #askbid["size"] = np.exp(askbid["size"])
        #askbid = askbid[askbid["size"] < 10]
        askbid = askbid[askbid["price"] < mid_price*1.01]
        askbid = askbid[askbid["price"] > mid_price*0.99]
        print(askbid)
        #askbid["price"] = np.round(askbid["price"]/10)*10
        #askbid["size"] = askbid["size"].groupby(askbid['price'].apply(lambda x: round(x, 1))).sum()

        fig, ax = plt.subplots(1, 1)
        for i, (key, group) in enumerate(askbid.groupby("type"), start=0):
            color = "red"
            if key == "asks":
                color = "blue"
            group.plot(kind='scatter', x=u'price', y=u'size', color=color, ax=ax, label=key)
        #plot = askbid.plot(kind='scatter', x="price", y="size")
        #fig = plot.get_figure()
        fig = ax.get_figure()
        fig.savefig("data/askbid.png")

    def get_portfolio(self):
        price = self.candles[-1, self.col["close"]]
        balance = self.exchange.getbalance()
        portfolio = {
            "asset": balance[self.asset]["amount"],
            "currency": balance[self.currency]["amount"],
            "price": price,
            "balance": False,
            }
        portfolio["balance"] = portfolio["currency"] + price * portfolio["asset"]
        return portfolio
    def get_best_bid_price(self):
        ticker = self.exchange.ticker()
        return ticker["best_bid"]
    def get_best_ask_price(self):
        ticker = self.exchange.ticker()
        return ticker["best_ask"]

    def dryrun(self):
        self.is_dryrun = True
        self.run()

    def run(self):
        self.get_hisotry()
        print(self.df)
        self.portfolio = self.get_portfolio()
        print(self.portfolio)
        self.backtest.set_df(self.df)
        self.backtest.resetbacktest()

        # pubnub watch
        message_callbacks = {
                    'lightning_executions_'+self.pair: self.receive_executions,
                    }
        self.realtimeexchange.watch(message_callbacks)

    def receive_executions(self, pubnub, message):
        executions = self.executions2df(message)
        if executions.shape[0] < 10:
            return
        print("History loaded. start trading")
        df = Backtest.resample_candle(executions, exchange="bitflyer", freq="%sN" % self.candle_nsec)
        df = pd.concat([self.df, df])
        df = df.drop_duplicates(subset='start', keep='first')
        self.df = df.iloc[:-1, :]
        self.backtest.action(self.df, self.strategy, self.params[self.strategy], self.step_action)

    def step_action(self, action):
        self.portfolio = self.get_portfolio()
        if action == BUY:
            if self.portfolio["currency"] == 0:
                return False
            price = self.get_best_bid_price()
            if not self.is_dryrun:
                self.order_id = self.exchange.buy(price, self.portfolio["currency"])
            self.portfolio = self.get_portfolio()
            print("BUY", self.portfolio)
            self.trades += 1
            return True
        if action == SELL:
            if self.portfolio["asset"] == 0:
                return False
            price = self.get_best_ask_price()
            if not self.is_dryrun:
                self.order_id = self.exchange.sell(price, self.portfolio["asset"])
            self.portfolio = self.get_portfolio()
            print("SELL", self.portfolio)
            self.trades += 1
            return True
        return False


    def executions2df(self, message):
        executions = pd.DataFrame.from_dict(message["message"])
        executions["timetoken"] = message["timetoken"]
        executions['exec_date'] = pd.to_datetime(executions['exec_date'])
        executions.index = executions["exec_date"]
        if self.executions is None:
            self.executions = executions
        else:
            self.executions = pd.concat([self.executions, executions])
        return self.executions

    def get_hisotry(self):
        candleSize = self.params[self.strategy]["candleSize"]
        historySize = self.params[self.strategy]["historySize"]
        self.candle_nsec = int(candleSize*6*10e9) # minute end nanosecond
        minutes = int(candleSize * historySize*10)
        from_date = datetime.datetime.today() - datetime.timedelta(minutes=minutes)
        trade = self.exchange.get_trade(from_date)
        candle = Backtest.resample_candle(trade, exchange="bitflyer", freq="%sN" % self.candle_nsec)
        candle.to_csv("data/realtime_bitflyer_%s_candle.csv" % self.pair, index=None)
        for col in "open,high,low,close,vwp,volume".split(","):
            candle[col] = candle[col].astype(np.float64)
        candle = candle.drop(candle.index[-1])
        self.df = candle
        self.size = self.df.shape[0]
        self.indsize = len(self.indicators)
        self.col = {x: self.df.columns.tolist().index(x) for x in ["open", "high", "low", "close", "vwp", "volume", "start"]}
        self.candles = self.df.values
        return self.df

    def report(self):
        dates = {
                "start": self.candles[0, self.col["start"]],
                "end": self.candles[-1, self.col["start"]],
                }
        timespan = dates["start"] - dates["end"]
        # the portfolio's balance is measured in {currency}
        startPrice = self.candles[0, self.col["close"]]
        endPrice = self.candles[-1, self.col["close"]]
        start = self.resetPortfolio()
        self.portfolio["balance"] = self.portfolio["currency"] + endPrice * self.portfolio["asset"]
        relativeProfit = (100 * self.portfolio["balance"] / start["balance"]) - 100
        profit = self.portfolio["balance"] - start["balance"]

        report = {
                "currency": self.portfolio["currency"],
                "asset": self.portfolio["asset"],

                "startTime": dates["start"].strftime('%Y-%m-%d %H:%M:%S'),
                "endTime": dates["end"].strftime('%Y-%m-%d %H:%M:%S'),
                "timespan": str(timespan),
                "market": endPrice * 100 / startPrice - 100,

                "balance": self.portfolio["balance"],
                "profit": profit,
                "relativeProfit": relativeProfit,

                #"yearlyProfit": round(profit / timespan.year()),
                #"relativeYearlyProfit": round(relativeProfit / timespan.year()),

                "startPrice": startPrice,
                "endPrice": endPrice,
                "trades": self.trades,
                "startBalance": start["balance"],
        }
        profitdf = self.df.copy()
        profitdf["balance"] = self.balance
        profitdf["relativeProfit"] = self.relativeProfit
        balance = profitdf["balance"].resample("7d").last().fillna(0).values
        close = profitdf["close"].resample("7d").last().fillna(0).values
        relativeProfit = profitdf["relativeProfit"].resample("7d").last().fillna(0).values

        report["alpha"] = report["relativeProfit"] - report["market"]
        report["sharp"] = self.sharp(relativeProfit)
        risk = test_risk_metrics(relativeProfit, close)
        risk2 = test_risk_adjusted_metrics(relativeProfit, close)
        report.update(risk)
        report.update(risk2)
        report["config"] = self.params[self.strategy]
        self.result.append(pd.DataFrame.from_dict([report]))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['run', 'dry-run', 'spread'])
    args = parser.parse_args()
    if args.mode == "run":
        realtime = Realtime()
        realtime.run()
    elif args.mode == "dry-run":
        realtime = Realtime()
        realtime.dryrun()
    elif args.mode == "spread":
        realtime = Realtime()
        realtime.spread()
    else:
        parser.print_help()
