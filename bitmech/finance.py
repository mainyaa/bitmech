#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://matsu911.github.io/org/quants_trading.html
# http://www.turingfinance.com/computational-investing-with-python-week-one/
# http://www.investopedia.com/articles/stocks/12/mitigating-downside-risk-with-sorentino-ratio.asp

import math
import numpy as np
import numpy.random as nrand

def vol(returns):
    # Return the standard deviation of returns
    return np.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = np.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    return np.cov(m)[0][1] / np.std(market)


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def var(returns, alpha):
    # This method calculates the historical simulation var of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)


def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    periods = np.min([periods, len(drawdowns)])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def average_dd_squared(returns, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    periods = np.min([periods, len(drawdowns)])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def treynor_ratio(er, returns, market, rf):
    return (er - rf) / beta(returns, market)

# 一定期間における超過リターン(無リスクレートを上回るリターン)の平均をリターンのバラツキで割ったシャープレシオはリスク調整済みリターンの典型例
def sharpe_ratio(er, returns, rf):
    return (er - rf) / vol(returns)

# シャープレシオの公式から無リスクレートを除去したものはインフォメーションレシオ
def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return np.abs(np.mean(diff) / vol(diff))


def modigliani_ratio(er, returns, benchmark, rf):
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(er, returns, rf, alpha):
    return (er - rf) / var(returns, alpha)


def conditional_sharpe_ratio(er, returns, rf, alpha):
    return (er - rf) / cvar(returns, alpha)

# オメガレシオ(正のリターンの合計/負のリターンの合計)
def omega_ratio(er, returns, rf, target=0):
    return (er - rf) / lpm(returns, target, 1)


def sortino_ratio(er, returns, rf, target=0):
    return (er - rf) / math.sqrt(lpm(returns, target, 2))


def kappa_three_ratio(er, returns, rf, target=0):
    return (er - rf) / math.pow(lpm(returns, target, 3), float(1/3))


def gain_loss_ratio(returns, target=0):
    return hpm(returns, target, 1) / lpm(returns, target, 1)


def upside_potential_ratio(returns, target=0):
    return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))

# カルマーレシオ(平均リターン/山から谷までの最大ドローダウン)
def calmar_ratio(er, returns, rf):
    return (er - rf) / max_dd(returns)

# スターリングレシオ(平均リターン/最大ドローダウンの平均)
def sterling_ratio(er, returns, rf, periods):
    return (er - rf) / average_dd(returns, periods)


def burke_ratio(er, returns, rf, periods):
    return (er - rf) / math.sqrt(average_dd_squared(returns, periods))

def test_risk_metrics(r=None, m=None):
    # This is just a testing method
    if r is None:
        r = nrand.uniform(-1, 1, 50)
    if m is None:
        m = nrand.uniform(-1, 1, 50)
    report = {
            "vol": vol(r),
            "beta": beta(r, m),
            "hpm(0.0)_1": hpm(r, 0.0, 1),
            "lpm(0.0)_1": lpm(r, 0.0, 1),
            "VaR(0.05)": var(r, 0.05),
            "CVaR(0.05)": cvar(r, 0.05),
            "Drawdown(5)": dd(r, 5),
            "Max Drawdown": max_dd(r),
            }
    return report


def test_risk_adjusted_metrics(r=None, m=None):
    # Returns from the portfolio (r) and market (m)
    if r is None:
        r = nrand.uniform(-1, 1, 50)
    if m is None:
        m = nrand.uniform(-1, 1, 50)
    # Expected return
    e = np.mean(r)
    # Risk free rate
    f = 0.05
    f = 0.065 / 12 / 30 # 10年国債
    # Risk-adjusted return based on Volatility
    report = {
            "Treynor Ratio": treynor_ratio(e, r, m, f),
            "Sharpe Ratio": sharpe_ratio(e, r, f),
            "Information Ratio": information_ratio(r, m),
            # Risk-adjusted return based on Value at Risk
            "Excess VaR": excess_var(e, r, f, 0.05),
            "Conditional Sharpe Ratio": conditional_sharpe_ratio(e, r, f, 0.05),
            # Risk-adjusted return based on Lower Partial Moments
            "Omega Ratio": omega_ratio(e, r, f),
            "Sortino Ratio": sortino_ratio(e, r, f),
            "Kappa 3 Ratio": kappa_three_ratio(e, r, f),
            "Gain Loss Ratio": gain_loss_ratio(r),
            "Upside Potential Ratio": upside_potential_ratio(r),
            # Risk-adjusted return based on Drawdown risk
            "Calmar Ratio": calmar_ratio(e, r, f),
            "Sterling Ratio": sterling_ratio(e, r, f, 5),
            "Burke Ratio": burke_ratio(e, r, f, 5),
    }
    return report


if __name__ == "__main__":
    print(test_risk_metrics())
    print(test_risk_adjusted_metrics())
