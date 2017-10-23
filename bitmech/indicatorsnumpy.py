#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def DEMA(df, n_fast, n_slow):
    shortEMA = moving_average(df, n_slow)
    longEMA = moving_average(df, n_fast)
    return 100 * (shortEMA - longEMA) / ((shortEMA + longEMA) / 2)

def MACD(df, n_fast, n_slow, signal=9):
    emaslow, emafast, emadiff = moving_average_convergence(df, nslow=n_slow, nfast=n_fast)
    signal = moving_average(emadiff, signal)
    macddiff = emadiff - signal
    return macddiff

def PPO(df, n_fast, n_slow, signal=9):
    emaslow, emafast, emadiff = moving_average_convergence(df, nslow=n_slow, nfast=n_fast)
    ppo = 100 * (emadiff / emaslow)
    PPOsignal = moving_average(ppo, signal)
    PPOhist = ppo - PPOsignal
    return PPOhist

def RSI(df, n):
    nint = int(n)

    deltas = np.diff(df)
    seed = deltas[:nint+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(df)
    rsi[:nint] = 100. - 100./(1.+rs)

    for i in range(nint, len(df)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def STOCHRSI(df, n):
    start = -int(n)
    if start < 0:
        start = 0
    rsi = RSI(df, n)
    RSIhistory = rsi[start:]
    minRSI = np.min(RSIhistory)
    maxRSI = np.max(RSIhistory)
    base = maxRSI - minRSI
    if base == 0:
        return 50.
    return ((rsi - minRSI) / base) * 100

def moving_average(x, span):
    if span < 1:
        span = 1
    return pd.stats.moments.ewma(x, span=span)

def moving_average_convergence(x, nslow, nfast):
    emaslow = moving_average(x, nslow)
    emafast = moving_average(x, nfast)
    return emaslow, emafast, emafast - emaslow
