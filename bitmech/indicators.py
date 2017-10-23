#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
# https://github.com/FreddieWitherden/ta/blob/master/ta.py

import pandas as pd
import numpy as np

#Moving Average
def MA(df, n):
    MA = pd.Series(pd.rolling_mean(df['close'], n), name = 'MA_' + str(n))
    return MA

#Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(df['close'].ewm(span = n, min_periods = int(n - 1)).mean(), name = 'EMA_' + str(n))
    return EMA

#Momentum
def MOM(df, n):
    M = pd.Series(df['close'].diff(n), name = 'Momentum_' + str(n))
    return M

#Rate of Change
def ROC(df, n):
    M = df['close'].diff(n - 1)
    N = df['close'].shift(n - 1)
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    return ROC

#Average True Range
def ATR(df, n):
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.ix[i + 1, 'high'], df.ix[i, 'close']) - min(df.ix[i + 1, 'low'], df.ix[i, 'close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span = n, min_periods = int(n)).mean(), name = 'ATR_' + str(n))
    return ATR

#Bollinger Bands
def BBANDS(df, n):
    MA = pd.Series(pd.rolling_mean(df['close'], n))
    MSD = pd.Series(pd.rolling_std(df['close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))
    return B1, B2

#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)
    R1 = pd.Series(2 * PP - df['low'])
    S1 = pd.Series(2 * PP - df['high'])
    R2 = pd.Series(PP + df['high'] - df['low'])
    S2 = pd.Series(PP - df['high'] + df['low'])
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    return PSR

#Stochastic oscillator %K
def STOK(df):
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')
    return SOk

#Stochastic oscillator %D
def STO(df, n):
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')
    SOd = pd.Series(SOk.ewm(span = n, min_periods = int(n - 1)).mean(), name = 'SO%d_' + str(n))
    return SOd

#Trix
def TRIX(df, n):
    EX1 = df['close'].ewm(span = n, min_periods = int(n - 1)).mean()
    EX2 = EX1.ewm(span = n, min_periods = int(n - 1)).mean()
    EX3 = EX2.ewm(span = n, min_periods = int(n - 1)).mean()
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    return Trix

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.ix[i + 1, 'high'] - df.ix[i, 'high']
        DoMove = df.ix[i, 'low'] - df.ix[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.ix[i + 1, 'high'], df.ix[i, 'close']) - min(df.ix[i + 1, 'low'], df.ix[i, 'close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span = n, min_periods = int(n)).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = int(n - 1) / ATR).mean())
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = int(n - 1) / ATR).mean())
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span = n_ADX, min_periods = int(n_ADX - 1)).mean(), name = 'ADX_' + str(n) + '_' + str(n_ADX))
    return ADX

#DEMA
def DEMA(df, n_fast, n_slow, signal=9):
    EMAfast = pd.Series(df['close'].ewm(span = n_fast, min_periods = int(n_slow - 1)).mean())
    EMAslow = pd.Series(df['close'].ewm(span = n_slow, min_periods = int(n_slow - 1)).mean())
    DEMA = pd.Series(100 * (EMAfast- EMAslow) / ((EMAfast + EMAslow) / 2), name = 'DEMA_' + str(n_fast) + '_' + str(n_slow))
    return DEMA

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow, signal=9):
    EMAfast = pd.Series(df['close'].ewm(span = n_fast, min_periods = int(n_slow - 1)).mean())
    EMAslow = pd.Series(df['close'].ewm(span = n_slow, min_periods = int(n_slow - 1)).mean())
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span = signal, min_periods = int(signal)).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    return MACD, MACDsign, MACDdiff

#PPO, PPO Signal and PPO difference
def PPO(df, n_fast, n_slow, signal=9):
    EMAfast = pd.Series(df['close'].ewm(span = n_fast, min_periods = int(n_slow - 1)).mean())
    EMAslow = pd.Series(df['close'].ewm(span = n_slow, min_periods = int(n_slow - 1)).mean())
    PPO = pd.Series(100 * (EMAfast - EMAslow) / EMAslow, name = 'PPO_' + str(n_fast) + '_' + str(n_slow))
    PPOsign = pd.Series(PPO.ewm(span = signal, min_periods = int(signal)).mean(), name = 'PPOsign_' + str(n_fast) + '_' + str(n_slow))
    PPOdiff = pd.Series(PPO - PPOsign, name = 'PPOdiff_' + str(n_fast) + '_' + str(n_slow))
    return PPO, PPOsign, PPOdiff

#Mass Index
def MassI(df):
    Range = df['high'] - df['low']
    EX1 = Range.ewm(span = 9, min_periods = 8).mean()
    EX2 = EX1.ewm(span = 9, min_periods = 8).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')
    return MassI

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def Vortex(df, n):
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.ix[i + 1, 'high'], df.ix[i, 'close']) - min(df.ix[i + 1, 'low'], df.ix[i, 'close'])
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.ix[i + 1, 'high'] - df.ix[i, 'low']) - abs(df.ix[i + 1, 'low'] - df.ix[i, 'high'])
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))
    return VI





#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['close'].diff(r1 - 1)
    N = df['close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['close'].diff(r2 - 1)
    N = df['close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['close'].diff(r3 - 1)
    N = df['close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['close'].diff(r4 - 1)
    N = df['close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    return KST

#Relative Strength Index
def RSI(df, n):
    UpMove = df['high'].diff(-1)
    DoMove = df['low'].diff(1)
    UpMove[UpMove <= DoMove] = 0
    DoMove[DoMove <= UpMove] = 0
    UpMove[UpMove < 0] = 0
    DoMove[DoMove < 0] = 0
    UpMove = pd.Series(UpMove)
    DoMove = pd.Series(DoMove)
    PosDI = pd.Series(UpMove.ewm(span = n, min_periods = int(n - 1)).mean())
    NegDI = pd.Series(DoMove.ewm(span = n, min_periods = int(n - 1)).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    return RSI

#Relative Strength Index
def RSI_(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 < df.shape[0]:
        UpMove = df.ix[i + 1, 'high'] - df.ix[i, 'high']
        DoMove = df.ix[i, 'low'] - df.ix[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = int(n - 1)).mean())
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = int(n - 1)).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    return RSI

#Relative Strength Index
def _RSI(df, n):
    nint = int(n)
    deltas = df["close"].diff()
    seed = deltas[:nint+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros((df.shape[0],), dtype=np.float64)
    rsi[:nint] = 100. - 100./(1.+rs)

    for i in range(nint, len(deltas)):
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
    RSI = pd.Series(rsi, name = 'RSI_' + str(n))
    return RSI

# Stochastic Relative Strength Index
def STOCHRSI(df, n):
    start = -int(n)
    if start < 0:
        start = 0
    rsi = RSI(df, n)
    RSIhistory = rsi[start:]
    minRSI = RSIhistory.min()
    maxRSI = RSIhistory.max()
    base = maxRSI - minRSI
    if base == 0:
        base = -minRSI
    STOCHRSI = pd.Series(((rsi - minRSI) / base), name = 'STOCHRSI_' + str(n))
    return STOCHRSI

#True Strength Index
def TSI(df, r, s):
    M = pd.Series(df['close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm(span = r, min_periods = int(r - 1)).mean())
    aEMA1 = pd.Series(aM.ewm(span = r, min_periods = int(r - 1)).mean())
    EMA2 = pd.Series(EMA1.ewm(span = s, min_periods = int(s - 1)).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span = s, min_periods = int(s - 1)).mean())
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))
    return TSI

#Accumulation/Distribution
def ACCDIST(df, n):
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    return AD

#Chaikin Oscillator
def Chaikin(df):
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    Chaikin = pd.Series(ad.ewm(span = 3, min_periods = 2).mean() - ad.ewm(span = 10, min_periods = 9).mean(), name = 'Chaikin')
    return Chaikin

#Money Flow Index and Ratio
def MFI(df, n):
    PP = (df['high'] + df['low'] + df['close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.ix[i + 1, 'volume'])
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))
    return MFI

#On-balance Volume
def OBV(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.ix[i + 1, 'close'] - df.ix[i, 'close'] > 0:
            OBV.append(df.ix[i + 1, 'volume'])
        if df.ix[i + 1, 'close'] - df.ix[i, 'close'] == 0:
            OBV.append(0)
        if df.ix[i + 1, 'close'] - df.ix[i, 'close'] < 0:
            OBV.append(-df.ix[i + 1, 'volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))
    return OBV_ma

#Force Index
def FORCE(df, n):
    F = pd.Series(df['close'].diff(n) * df['volume'].diff(n), name = 'Force_' + str(n))
    return F

#Ease of Movement
def EOM(df, n):
    EoM = (df['high'].diff(1) + df['low'].diff(1)) * (df['high'] - df['low']) / (2 * df['volume'])
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))
    return Eom_ma

#Commodity Channel Index
def CCI(df, n):
    PP = (df['high'] + df['low'] + df['close']) / 3
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))
    return CCI

#Coppock Curve
def COPP(df, n):
    M = df['close'].diff(int(n * 11 / 10) - 1)
    N = df['close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['close'].diff(int(n * 14 / 10) - 1)
    N = df['close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series((ROC1 + ROC2).ewm(span = n, min_periods = int(n)).mean(), name = 'Copp_' + str(n))
    return Copp

#Keltner Channel
def KELCH(df, n):
    KelChM = pd.Series(pd.rolling_mean((df['high'] + df['low'] + df['close']) / 3, n), name = 'KelChM_' + str(n))
    KelChU = pd.Series(pd.rolling_mean((4 * df['high'] - 2 * df['low'] + df['close']) / 3, n), name = 'KelChU_' + str(n))
    KelChD = pd.Series(pd.rolling_mean((-2 * df['high'] + 4 * df['low'] + df['close']) / 3, n), name = 'KelChD_' + str(n))
    return KelChM, KelChU, KelChD

#Ultimate Oscillator
def ULTOSC(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.ix[i + 1, 'high'], df.ix[i, 'close']) - min(df.ix[i + 1, 'low'], df.ix[i, 'close'])
        TR_l.append(TR)
        BP = df.ix[i + 1, 'close'] - min(df.ix[i + 1, 'low'], df.ix[i, 'close'])
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')
    return UltO

#Donchian Channel
def DONCH(df, n):
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['high'].ix[i:i + n - 1]) - min(df['low'].ix[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    return DonCh

#Standard Deviation
def STDDEV(df, n):
    return pd.Series(pd.rolling_std(df['close'], n), name = 'STD_' + str(n))




from functools import wraps

from pandas import DataFrame, Series
from pandas.stats import moments


def series_indicator(col):
    def inner_series_indicator(f):
        @wraps(f)
        def wrapper(s, *args, **kwargs):
            if isinstance(s, DataFrame):
                s = s[col]
            return f(s, *args, **kwargs)
        return wrapper
    return inner_series_indicator


def _wilder_sum(s, n):
    s = s.dropna()

    nf = (n - 1) / n
    ws = [np.nan]*(n - 1) + [s[n - 1] + nf*sum(s[:n - 1])]

    for v in s[n:]:
        ws.append(v + ws[-1]*nf)

    return Series(ws, index=s.index)


@series_indicator('high')
def hhv(s, n):
    return moments.rolling_max(s, n)


@series_indicator('low')
def llv(s, n):
    return moments.rolling_min(s, n)


@series_indicator('close')
def ema(s, n, wilder=False):
    span = n if not wilder else 2*n - 1
    return moments.ewma(s, span=span)


@series_indicator('close')
def macd(s, nfast=12, nslow=26, nsig=9, percent=True):
    fast, slow = ema(s, nfast), ema(s, nslow)

    if percent:
        macd = 100*(fast / slow - 1)
    else:
        macd = fast - slow

    sig = ema(macd, nsig)
    hist = macd - sig

    return DataFrame(dict(macd=macd, signal=sig, hist=hist,
                          fast=fast, slow=slow))


def aroon(s, n=25):
    up = 100 * moments.rolling_apply(s.high, n + 1, lambda x: x.argmax()) / n
    dn = 100 * moments.rolling_apply(s.low, n + 1, lambda x: x.argmin()) / n

    return DataFrame(dict(up=up, down=dn))


@series_indicator('close')
def rsi(s, n=14):
    diff = s.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n, wilder=True)
    emadn = ema(dn, n, wilder=True)

    return 100 * emaup/(emaup + emadn)


def stoch(s, nfastk=14, nfullk=3, nfulld=3):
    if not isinstance(s, DataFrame):
        s = DataFrame(dict(high=s, low=s, close=s))

    hmax, lmin = hhv(s, nfastk), llv(s, nfastk)

    fastk = 100 * (s.close - lmin)/(hmax - lmin)
    fullk = moments.rolling_mean(fastk, nfullk)
    fulld = moments.rolling_mean(fullk, nfulld)

    return DataFrame(dict(fastk=fastk, fullk=fullk, fulld=fulld))


@series_indicator('close')
def dtosc(s, nrsi=13, nfastk=8, nfullk=5, nfulld=3):
    srsi = stoch(rsi(s, nrsi), nfastk, nfullk, nfulld)
    return DataFrame(dict(fast=srsi.fullk, slow=srsi.fulld))


def atr(s, n=14):
    cs = s.close.shift(1)
    tr = s.high.combine(cs, max) - s.low.combine(cs, min)

    return ema(tr, n, wilder=True)


def cci(s, n=20, c=0.015):
    if isinstance(s, DataFrame):
        s = s[['high', 'low', 'close']].mean(axis=1)

    mavg = moments.rolling_mean(s, n)
    mdev = moments.rolling_apply(s, n, lambda x: np.fabs(x - x.mean()).mean())

    return (s - mavg)/(c * mdev)


def cmf(s, n=20):
    clv = (2*s.close - s.high - s.low) / (s.high - s.low)
    vol = s.volume

    return moments.rolling_sum(clv*vol, n) / moments.rolling_sum(vol, n)


def force(s, n=2):
    return ema(s.close.diff()*s.volume, n)


@series_indicator('close')
def kst(s, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9):
    rocma1 = moments.rolling_mean(s / s.shift(r1) - 1, n1)
    rocma2 = moments.rolling_mean(s / s.shift(r2) - 1, n2)
    rocma3 = moments.rolling_mean(s / s.shift(r3) - 1, n3)
    rocma4 = moments.rolling_mean(s / s.shift(r4) - 1, n4)

    kst = 100*(rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4)
    sig = moments.rolling_mean(kst, nsig)

    return DataFrame(dict(kst=kst, signal=sig))


def ichimoku(s, n1=9, n2=26, n3=52):
    conv = (hhv(s, n1) + llv(s, n1)) / 2
    base = (hhv(s, n2) + llv(s, n2)) / 2

    spana = (conv + base) / 2
    spanb = (hhv(s, n3) + llv(s, n3)) / 2

    return DataFrame(dict(conv=conv, base=base, spana=spana.shift(n2),
                          spanb=spanb.shift(n2), lspan=s.close.shift(-n2)))


def ultimate(s, n1=7, n2=14, n3=28):
    cs = s.close.shift(1)
    bp = s.close - s.low.combine(cs, min)
    tr = s.high.combine(cs, max) - s.low.combine(cs, min)

    avg1 = moments.rolling_sum(bp, n1) / moments.rolling_sum(tr, n1)
    avg2 = moments.rolling_sum(bp, n2) / moments.rolling_sum(tr, n2)
    avg3 = moments.rolling_sum(bp, n3) / moments.rolling_sum(tr, n3)

    return 100*(4*avg1 + 2*avg2 + avg3) / 7


def auto_envelope(s, nema=22, nsmooth=100, ndev=2.7):
    sema = ema(s.close, nema)
    mdiff = s[['high','low']].sub(sema, axis=0).abs().max(axis=1)
    csize = moments.ewmstd(mdiff, nsmooth)*ndev

    return DataFrame(dict(ema=sema, lenv=sema - csize, henv=sema + csize))


@series_indicator('close')
def bbands(s, n=20, ndev=2):
    mavg = moments.rolling_mean(s, n)
    mstd = moments.rolling_std(s, n)

    hband = mavg + ndev*mstd
    lband = mavg - ndev*mstd

    return DataFrame(dict(ma=mavg, lband=lband, hband=hband))


def safezone(s, position, nmean=10, npen=2.0, nagg=3):
    if isinstance(s, DataFrame):
        s = s.low if position == 'long' else s.high

    sgn = -1.0 if position == 'long' else 1.0

    # Compute the average upside/downside penetration
    pen = moments.rolling_apply(
        sgn*s.diff(), nmean,
        lambda x: x[x > 0].mean() if (x > 0).any() else 0
    )

    stop = s + sgn*npen*pen
    return hhv(stop, nagg) if position == 'long' else llv(stop, nagg)


def sar(s, af=0.02, amax=0.2):
    high, low = s.high, s.low

    # Starting values
    sig0, xpt0, af0 = True, high[0], af
    sar = [low[0] - (high - low).std()]

    for i in xrange(1, len(s)):
        sig1, xpt1, af1 = sig0, xpt0, af0

        lmin = min(low[i - 1], low[i])
        lmax = max(high[i - 1], high[i])

        if sig1:
            sig0 = low[i] > sar[-1]
            xpt0 = max(lmax, xpt1)
        else:
            sig0 = high[i] >= sar[-1]
            xpt0 = min(lmin, xpt1)

        if sig0 == sig1:
            sari = sar[-1] + (xpt1 - sar[-1])*af1
            af0 = min(amax, af1 + af)

            if sig0:
                af0 = af0 if xpt0 > xpt1 else af1
                sari = min(sari, lmin)
            else:
                af0 = af0 if xpt0 < xpt1 else af1
                sari = max(sari, lmax)
        else:
            af0 = af
            sari = xpt0

        sar.append(sari)

    return Series(sar, index=s.index)


def adx(s, n=14):
    cs = s.close.shift(1)
    tr = s.high.combine(cs, max) - s.low.combine(cs, min)
    trs = _wilder_sum(tr, n)

    up = s.high - s.high.shift(1)
    dn = s.low.shift(1) - s.low

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn

    dip = 100 * _wilder_sum(pos, n) / trs
    din = 100 * _wilder_sum(neg, n) / trs

    dx = 100 * np.abs((dip - din)/(dip + din))
    adx = ema(dx, n, wilder=True)

    return DataFrame(dict(adx=adx, dip=dip, din=din))


def chandelier(s, position, n=22, npen=3):
    if position == 'long':
        return hhv(s, n) - npen*atr(s, n)
    else:
        return llv(s, n) + npen*atr(s, n)


def vortex(s, n=14):
    ss = s.shift(1)

    tr = s.high.combine(ss.close, max) - s.low.combine(ss.close, min)
    trn = moments.rolling_sum(tr, n)

    vmp = np.abs(s.high - ss.low)
    vmm = np.abs(s.low - ss.high)

    vip = moments.rolling_sum(vmp, n) / trn
    vin = moments.rolling_sum(vmm, n) / trn

    return DataFrame(dict(vin=vin, vip=vip))


@series_indicator('close')
def gmma(s, nshort=[3, 5, 8, 10, 12, 15],
         nlong=[30, 35, 40, 45, 50, 60]):
    short = {str(n): ema(s, n) for n in nshort}
    long = {str(n): ema(s, n) for n in nlong}

    return DataFrame(short), DataFrame(long)


def zigzag(s, pct=5):
    ut = 1 + pct / 100
    dt = 1 - pct / 100

    ld = s.index[0]
    lp = s.close[ld]
    tr = None

    zzd, zzp = [ld], [lp]

    for ix, ch, cl in zip(s.index, s.high, s.low):
        # No initial trend
        if tr is None:
            if ch / lp > ut:
                tr = 1
            elif cl / lp < dt:
                tr = -1
        # Trend is up
        elif tr == 1:
            # New high
            if ch > lp:
                ld, lp = ix, ch
            # Reversal
            elif cl / lp < dt:
                zzd.append(ld)
                zzp.append(lp)

                tr, ld, lp = -1, ix, cl
        # Trend is down
        else:
            # New low
            if cl < lp:
                ld, lp = ix, cl
            # Reversal
            elif ch / lp > ut:
                zzd.append(ld)
                zzp.append(lp)

                tr, ld, lp = 1, ix, ch

    # Extrapolate the current trend
    if zzd[-1] != s.index[-1]:
        zzd.append(s.index[-1])

        if tr is None:
            zzp.append(s.close[zzd[-1]])
        elif tr == 1:
            zzp.append(s.high[zzd[-1]])
        else:
            zzp.append(s.low[zzd[-1]])

    return Series(zzp, index=zzd)

