#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hyperopt import hp

def get_setting():
    return {
            "realtime": {
                    "exchange": "bitflyer",
                    "pair": "BTC_JPY",
                    "strategy": "STOCHRSI",
                    },

            "coincheck": {
                    "api_key": "",
                    "api_secret": "",
                    },
            "bitflyer": {
                    "api_key": "",
                    "api_secret": "",
                    },
            "indicator": ["MACD", "PPO", "RSI", "STOCHRSI"],
            "indicatorsize": 6,
            "CCI": {
                    "constant": 0.015,
                    "history": 90,
                    "thresholds": {
                            "down": -100,
                            "persistence": 0,
                            "up": 100
                            }
                    },
            # output_top/poloniex_USDT_BTC_DEMA_20170920_080544_78.82873409_response.json
            "DEMA": {
                    "candleSize": 2.9726010636843285,
                    "historySize": 17.303705527590196,
                    "n_fast": 29.596405665443594,
                    "n_slow": 99,
                    "thresholds": {
                            "down": -0.2954274144265062,
                            "up": 0.4469963291742115
                            },
                    },
            "I understand that Bitmech only automates MY OWN trading strategies": False,
            # output_top/poloniex_USDT_BTC_MACD_20170920_100452_80.8675734831_response.json
            "MACD": {
                    "candleSize": 30.84060197313068,
                    "historySize": 12.96831852017042,
                    "n_fast": 29.99804605367707,
                    "n_slow": 37,
                    "signal": 6.649309931575975,
                    "thresholds": {
                            "down": -3.6157208883405954,
                            "up": 4.2036548218718455,
                            "persistence": 8
                            },
                    },
            # output_top/poloniex_USDT_BTC_PPO_20170920_120830_12.8130676685_response.json
            "PPO": {
                    'candleSize': 15.012211063006566,
                    'historySize': 11.276191356255557,
                    'n_slow': 55.003781290395544,
                    'n_fast': 13.19870720809385,
                    'signal': 11.86271190497146,
                    'thresholds': {
                            'down': -0.7283493088128622,
                            'persistence': 14,
                            'up': 0.5086717469381983
                            }
                    },
            "PPO_RiskVariance": {
                    'candleSize': 15.008512033798747,
                    'historySize': 28.452699825700602,
                    'n_slow': 127.77332863155551,
                    'n_fast': 28.67705151903631,
                    'signal': 12.101330890666896,
                    'thresholds': {
                            'down': -0.3152518867558186,
                            'persistence': 46,
                            'up': 0.6485866186319462
                            }
                    },
            # output_21_100/poloniex_USDT_BTC_RSI_20170922_232604_15.7489142195_response.json
            "RSI": {
                    'candleSize': 16.43056564093968,
                    'historySize': 57.47305213100427,
                    'n': 15.49907592052065,
                    'thresholds': {
                            'high': 52.541618301990226,
                            'low': 25.985752816105094,
                            'persistence': 48
                            },
                    },
            # output_21_200/poloniex_USDT_BTC_StochRSI_20170924_052932_26.7465182926_response.json
            "STOCHRSI": {
                    'candleSize': 10.909908365090716,
                    'historySize': 19.054914036946773,
                    "n": 11.089642549853192,
                    'thresholds': {
                            'high': 69.72180508034852,
                            'low': 15.545354389349995,
                            'persistence': 5}
                    },
            "TSI": {
                    "n_slow": 25,
                    "n_fast": 13,
                    "thresholds": {
                            "high": 25,
                            "low": -25,
                            "persistence": 1
                            }
                    },
            "UO": {
                    "candleSize": 6.242617586441378,
                    "historySize": 53.93949091401692,
                    "first": {
                            "weight": 7.288936046254792,
                            "period": 9.998980451884936
                            },
                    "second": {
                            "weight": 3.9548291908742677,
                            "period": 9.475841380726544
                            },
                    "third": {
                            "weight": 1.702352989203846,
                            "period": 19.05339243453129
                            },
                    "thresholds": {
                            "low": 22.03485534183639,
                            "high": 66.69548975267158,
                            "persistence": 47
                            }
                    },
            'hyperopt': {'deltaDays': 30,
                        'testDays': 90,
                        'num_rounds': 100,
                        'random_state': 2017,
                        'num_iter': 50,
                        'init_points': 90,
                        'parallel': False,
                        'Strategy': 'PPO',
                        'show_chart': False,
                        'save': True,
                        'watch':{
                                "exchange": "poloniex",
                                "currency": 'USDT',
                                "asset": 'BTC'
                        },
                        "DEMA":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "n_fast": hp.uniform("n_fast", 1.,30), # n_fast EMA
                               "n_slowWeight": hp.uniform("n_slowWeight", 1.,5.), # n_slow EMA(n_fast*n_slowWeight)
                               "thresholds-down": hp.uniform("thresholds-down", -0.5,0.), # trend thresholds
                               "thresholds-up": hp.uniform("thresholds-up", 0.,0.5), # trend thresholds
                        },
                        "MACD":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "n_fast": hp.uniform("n_fast", 1.,30), # n_fast EMA
                               "n_slowWeight": hp.uniform("n_slowWeight", 1.,5.), # n_slow EMA(n_fast*n_slowWeight)
                               "signal": hp.uniform("signal", 1,18), # n_fastEMA - n_slowEMA diff
                               "thresholds-down": hp.uniform("thresholds-down", -5.,0.), # trend thresholds
                               "thresholds-up": hp.uniform("thresholds-up", 0.,5.), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "PPO":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "n_fast": hp.uniform("n_fast", 1.,30), # n_fast EMA
                               "n_slowWeight": hp.uniform("n_slowWeight", 1.,5.), # n_slow EMA(n_fast*n_slowWeight)
                               "signal": hp.uniform("signal", 1,18), # 100 * (n_fastEMA - n_slowEMA / n_slowEMA)
                               "thresholds-down": hp.uniform("thresholds-down", -5.,0.), # trend thresholds
                               "thresholds-up": hp.uniform("thresholds-up", 0.,5.), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        # Uses one of the momentum indicators but adjusts the thresholds when PPO is bullish or bearish
                        # Uses settings from the ppo and momentum indicator config block
                        "varPPO":{ # TODO: merge PPO config
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "n_fast": hp.uniform("n_fast", 1.,30), # n_fast EMA
                               "n_slowWeight": hp.uniform("n_slowWeight", 1.,5.), # n_slow EMA(n_fast*n_slowWeight)
                               "signal": hp.uniform("signal", 1,18), # 100 * (n_fastEMA - n_slowEMA / n_slowEMA)
                               "thresholds-down": hp.uniform("thresholds-down", -5.,0.), # trend thresholds
                               "thresholds-up": hp.uniform("thresholds-up", 0.,5.), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                               "momentum": hp.uniform("momentum", 0, 2.99999), # index of ["RSI", "TSI", "UO"]
                               # new threshold is default threshold + PPOhist * PPOweight
                               "weightLow": hp.uniform("weightLow", 60, 180),
                               "weightHigh": hp.uniform("weightHigh", -60, -180),
                        },
                        "RSI":{
                               "candleSize": hp.uniform("candleSize", 16.430566,18.252666), # tick per day
                               "historySize": hp.uniform("historySize", 56.089028,57.907360),
                               "n": hp.uniform("n", 15.358212,16.541936), # weight
                               "thresholds-low": hp.uniform("thresholds-low", 25.728315,26.203461), # trend thresholds
                               "thresholds-high": hp.uniform("thresholds-high", 52.276421,52.656804), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 45,49), # trend duration(count up by tick) thresholds
                        },
                        "STOCHRSI":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "history": hp.uniform("history", 7,21), # weight
                               "n": hp.uniform("n", 7,21), # weight
                               "thresholds-low": hp.uniform("thresholds-low", 0,50), # trend thresholds
                               "thresholds-high": hp.uniform("thresholds-high", 50.,100), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "CCI":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "consistant": hp.uniform("consistant", 7,21), # constant multiplier. 0.015 gets to around 70% fit
                               "history": hp.uniform("history", 45,135), # history size, make same or smaller than history
                               "thresholds-down": hp.uniform("thresholds-down", -50,-150), # trend thresholds
                               "thresholds-up": hp.uniform("thresholds-up", 50,150), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "TSI":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "n_fast": hp.uniform("n_fast", 1.,30), # n_fast EMA
                               "n_slowWeight": hp.uniform("n_slowWeight", 1.,5.), # n_slow EMA(n_fast*n_slowWeight)
                               "thresholds-low": hp.uniform("thresholds-low", -13,-52), # trend thresholds
                               "thresholds-high": hp.uniform("thresholds-high", 13,52), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                         },
                        "gannswing":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "vixperiod": hp.uniform("vixperiod", 1,80), # 
                               "swingperiod": hp.uniform("swingperiod", 1,40), # 
                               "stoploss-enabled": hp.uniform("stoploss-enabled", 0,1), # 
                               "stoploss-trailing": hp.uniform("stoploss-trailing", 0,1), # 
                               "stoploss-percent": hp.uniform("stoploss-percent", 0,5), # 
                         },
                        "UO":{
                               "candleSize": hp.uniform("candleSize", 1,20), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "first-weight": hp.uniform("first-weight", 2,8), # 
                               "first-period": hp.uniform("first-period", 4.5,14), # 
                               "second-weight": hp.uniform("second-weight", 1,4), # 
                               "second-period": hp.uniform("second-period", 7,28), # 
                               "third-weight": hp.uniform("third-weight", 0.5,2), # 
                               "third-period": hp.uniform("third-period", 14,56), # 
                               "thresholds-low": hp.uniform("thresholds-low", 15,45), # trend thresholds
                               "thresholds-high": hp.uniform("thresholds-high", 45,140), # trend thresholds
                               "thresholds-persistence": hp.uniform("thresholds-persistence", 0,100), # trend duration(count up by tick) thresholds
                         },
            },
        }
