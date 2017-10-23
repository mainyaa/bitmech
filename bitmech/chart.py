#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from plotInfo import plotEvolutionSummary


def moving_average(x, n, weight_type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    n = int(n)
    x = np.asarray(x)
    if weight_type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http:#stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http:#www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def moving_average_convergence(x, nslow=26, nfast=12, weight_type='exponential'):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, weight_type=weight_type)
    emafast = moving_average(x, nfast, weight_type=weight_type)
    return emaslow, emafast, emafast - emaslow

def candlechart(fig, ax, ohlc, width=0.8):
    return mpf.candlestick2_ohlc(ax, opens=ohlc.open.values, closes=ohlc.close.values,
                          lows=ohlc.low.values, highs=ohlc.high.values,
                          width=width, colorup='r', colordown='b')

def dema(ax, price, params):
    emaslow, emafast, emadiff = moving_average_convergence(price, nslow=params["long"], nfast=params["short"])
    vstack = np.vstack((range(len(emaslow)), emaslow.T)).T
    ax.plot(vstack[:, 0], vstack[:, 1], label="Close EMA({})".format(params["long"]))
    vstack = np.vstack((range(len(emafast)), emafast.T)).T
    ax.plot(vstack[:, 0], vstack[:, 1], label="Close EMA({})".format(params["short"]))
    return ax

def macd(fig, ax, candles, params):
    axis = candles["start"].values
    price = candles["close"].values
    emaslow, emafast, emadiff = moving_average_convergence(price, nslow=params["long"], nfast=params["short"])
    signal = moving_average_convergence(emadiff, nslow=params["long"], nfast=params["short"])
    signal = moving_average(emadiff, params["signal"], weight_type='exponential')
    signal = emadiff - signal
    print(params)
    print(signal.shape)
    print(candles.shape)
    trades = macd_trade(signal, candles, params)
    print(trades)
    # trade
    if trades.shape[0] > 0:
        trade(ax, trades, candles, params)
    # divide graph
    vstack = np.vstack((range(len(signal)), signal.T)).T
    divider = make_axes_locatable(ax)
    #ax_bot = divider.append_axes("bottom", size="25%", pad=0., sharex=ax)
    ax_bot = divider.append_axes("bottom", size="25%", pad=0.)
    fig.add_axes(ax_bot)
    ax_bot.plot(axis, vstack[:, 1], label="signal({})".format(params["signal"]))
    ax_bot.hlines([params["thresholds"]["down"], params["thresholds"]["up"]], axis[0], axis[-1], linestyles="dashed", label="thresholds({},{})".format(params["thresholds"]["down"],params["thresholds"]["up"]))
    ax_bot.grid()
    return ax_bot

def macd_trade(results, candles, params):
    trend = {
        "duration": 0,
        "persisted": False,
        "direction": 'none',
        "adviced": False
    }
    trades = []
    date_col = candles.columns.index("date")
    price_col = candles.columns.index("close")
    c = candles.values
    for i in range(c.shape[0]):
        macddiff = results[i]
        date = c[i,date_col]
        price = c[i,price_col]

        if macddiff > params["thresholds"]["up"]:

            # new trend detected
            if trend["direction"] != 'up':
                # reset the state for the new trend
                trend = {
                    "duration": 0,
                    "persisted": False,
                    "direction": 'up',
                    "adviced": False
                }

            trend["duration"] += 1

            print('In uptrend since', trend["duration"], 'candle(s)')

            if trend["duration"] >= params["thresholds"]["persistence"]:
                trend["persisted"] = True

            if trend["persisted"] and not trend["adviced"]:
                trend["adviced"] = True
                print("advice('long')")
                t = pd.DataFrame([[date, "buy", price]], columns=["date", "action", "price"])
                t.index = t["date"]
                trades.append(t)

        elif macddiff < params["thresholds"]["down"]:

            # new trend detected
            if trend["direction"] != 'down':
                # reset the state for the new trend
                trend = {
                    "duration": 0,
                    "persisted": False,
                    "direction": 'down',
                    "adviced": False
                }

            trend["duration"]+=1

            print('In downtrend since', trend["duration"], 'candle(s)')

            if trend["duration"] >= params["thresholds"]["persistence"]:
                trend["persisted"] = True

            if trend["persisted"] and not trend["adviced"]:
                trend["adviced"] = True
                print("advice('short')")
                t = pd.DataFrame([[date, "sell", price]], columns=["date", "action", "price"])
                t.index = t["date"]
                print(t)
                trades.append(t)
    print(trades)
    trades = pd.concat(trades)
    print(trades)
    return trades

def ppo(fig, ax, axis, price, params):
    emaslow, emafast, emadiff = moving_average_convergence(price, nslow=params["long"], nfast=params["short"])
    signal = moving_average_convergence(emadiff, nslow=params["long"], nfast=params["short"])
    PPOsignal = 100 * (emadiff / emaslow)
    PPOsignal = moving_average(PPOsignal, params["signal"], weight_type='exponential')
    signal = emadiff - signal

    # divide graph
    divider = make_axes_locatable(ax)
    #ax_bot = divider.append_axes("bottom", size="25%", pad=0., sharex=ax)
    ax_bot = divider.append_axes("bottom", size="25%", pad=0.)
    fig.add_axes(ax_bot)

    vstack = np.vstack((range(len(PPOsignal)), PPOsignal.T)).T
    ax_bot.plot(axis, vstack[:, 1], label="PPO signal({})".format(params["signal"]))
    #ax_bot.hlines([params["thresholds"]["down"], params["thresholds"]["up"]], axis[0], axis[-1], linestyles="dashed", label="thresholds({},{})".format(params["thresholds"]["down"],params["thresholds"]["up"]))
    ax_bot.grid()
    return ax_bot

def trade(ax, trades, candles, params):
    trades["start"] = pd.to_datetime(trades["date"])
    trades.index = trades["start"]
    buys = trades.loc[trades.action.str.match("buy"), :]
    buys = candles.merge(buys, on="start", how="left")
    sells = trades.loc[trades.action.str.match("sell"), :]
    sells = candles.merge(sells, on="start", how="left")

    ax.plot(buys.index, buys['price'], '^', markersize=8, color='g', label="buy")
    ax.plot(sells.index, sells['price'], 'v', markersize=8, color='g', label="sell")

def ohlcsum(df):
    return {
       'open': df['open'][0],
       'high': df['high'].max(),
       'low': df['low'].min(),
       'close': df['close'][-1],
       'volume': df['volume'].sum()
      }
def groupby_ohlc(candles, freq):
    return candles.groupby(pd.Grouper(freq=freq)).agg(ohlcsum)

def show_candles(candles, trades=None, params=None, candle_freq=None):
    strategy = "MACD"
    import config
    setting = config.get_setting()
    params = setting[strategy]

    # candle
    fig = plt.figure(1)
    ax = plt.subplot(1, 1, 1)
    if candle_freq:
        candlechart(fig, ax, candles)
        ohlc = groupby_ohlc(candles, freq=candle_freq)
        ax2 = ax.twiny()
        candlechart(fig, ax2, ohlc)
        ax2.autoscale()
        ax.autoscale()
    else:
        candlechart(fig, ax, candles)
        #ax.autoscale()

    # trade
    if trades != None and len(trades) > 0:
        trade(ax, trades, candles, params)

    # DEMA
    if strategy == "DEMA" and params != None:
        dema(ax, candles["close"].values, params)
    # MACD
    if strategy == "MACD" and params != None:
        params= {
                "short": 10,
                "long": 21,
                "signal": 9,
                "thresholds": {
                        "down": -0.025,
                        "up": 0.025,
                        "persistence": 1
                        }
        }
        macd(fig, ax, candles, params)
    # PPO
    if strategy == "PPO" and params != None:
        ppo(fig, ax, candles["start"].values, candles["close"].values, params)

    # show
    ax.grid()
    fig.autofmt_xdate()
    plt.savefig(strategy+"_bitflyer_candle.png")
    print(strategy+"_bitflyer_candle.png")
    import os
    os.system("open "+strategy+"_bitflyer_candle.png")


def show_chart():
    import backtest
    import datetime
    candle = backtest.Backtest.load_bitflyer()
    print(candle.iloc[-1,0])
    candle = backtest.Backtest.daterange(candle, start=candle.iloc[-1,0] - datetime.timedelta(days=1), end=candle.iloc[-1,0])
    candle = backtest.Backtest.resample_ohlc(candle, "10min")
    show_candles(candle)

if __name__ == '__main__':
    show_chart()
