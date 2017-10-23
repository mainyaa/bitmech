from hyperopt import hp

def get_setting():
    return {
            "coincheck": {
                    "api_key": "",
                    "api_secret": "",
                    },
            "bitflyer": {
                    "api_key": "",
                    "api_secret": "",
                    },
            "indicators": ["DEMA", "MACD", "PPO", "RSI", "STOCHRSI"],
            "indicatorsize": 5,
            "CCI": {
                    "constant": 0.015,
                    "history": 90,
                    "thresholds": {
                            "down": -100,
                            "persistence": 0,
                            "up": 100
                            }
                    },
            # output_21/300/poloniex_USDT_BTC_DEMA_20171008_124556_12.1648409665_config.json
            "DEMA": {
                    "candleSize": 57.71608611651829,
                    "historySize": 45.84586511154581,
                    "short": 11.047468650301287,
                    "long": 13.521498089419792,
                    "thresholds": {
                            "down": -0.22062923923705702,
                            "up": 0.2971497668786547
                            },
                    },
            "I understand that Bitmech only automates MY OWN trading strategies": False,
            # output_21_300/poloniex_USDT_BTC_MACD_20171002_043512_16.1096798321_config.json
            "MACD": {
                    "candleSize": 50.37603262287613,
                    "historySize": 49.11385249240558,
                    "short": 11.190538130227413,
                    "long": 31.9361776661688,
                    "signal": 8.345598472515947,
                    "thresholds": {
                            "down": -2.647355926611602,
                            "up": 4.830104575041095,
                            "persistence": 25
                            },
                    },
            # output_21_300/poloniex_USDT_BTC_PPO_20171002_100829_11.6076027373_config.json
            "PPO": {
                    "candleSize": 53.11970872385835,
                    "historySize": 14.324952603907466,
                    "short": 17.473854214347806,
                    "long": 84.56322009055627,
                    "signal": 8.807257152663537,
                    "thresholds": {
                            "down": -0.21076603917426873,
                            "up": 2.851268102378877,
                            "persistence": 30
                            }
                    },
            # output_21_300/poloniex_USDT_BTC_RSI_20171002_174916_13.4161424664_config.json
            "RSI": {
                    "candleSize": 37.23583684941627,
                    "historySize": 30.589514194439374,
                    "interval": 18.030527529418386,
                    "thresholds": {
                            "low": 0.10426461392160635,
                            "high": 64.90499053943151,
                            "persistence": 96
                            }
                    },
            # output_21_300/poloniex_USDT_BTC_StochRSI_20171002_143135_23.9282294228_config.json
            "STOCHRSI": {
                    "candleSize": 39.756808477380076,
                    "historySize": 1.8696973929673688,
                    "interval": 7.233447728599271,
                    "thresholds": {
                            "low": 13.602735892710804,
                            "high": 74.46356460866173,
                            "persistence": 6
                            }
                    },
            "TSI": {
                    "long": 25,
                    "short": 13,
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
            "adapter": "sqlite",
            "adviceLogger": {
                    "enabled": False,
                    "muteSoft": True
                    },
            "adviceWriter": {
                    "enabled": False,
                    "muteSoft": True
                    },
            "backtest": {
                    "batchSize": 50,
                    "daterange": "scan"
                    },
            "campfire": {
                    "account": "",
                    "apiKey": "",
                    "emitUpdates": False,
                    "enabled": False,
                    "nickname": "Gordon",
                    "roomId": None
                    },
            "candleWriter": {
                    "enabled": False
                    },
            "custom": {
                    "my_custom_setting": 10
                    },
            "debug": True,
            "importer": {
                    "daterange": {
                            "from": "2016-01-01 00:00:00"
                            }
                    },
            "ircbot": {
                    "botName": "gekkobot",
                    "channel": "#your-channel",
                    "emitUpdates": False,
                    "enabled": False,
                    "muteSoft": True,
                    "server": "irc.freenode.net"
                    },
            "mailer": {
                    "email": "",
                    "enabled": False,
                    "from": "",
                    "muteSoft": True,
                    "password": "",
                    "port": "",
                    "sendMailOnStart": True,
                    "server": "smtp.gmail.com",
                    "smtpauth": True,
                    "ssl": True,
                    "tag": "[Bitmech] ",
                    "to": "",
                    "user": ""
                    },
            "mongodb": {
                    "connectionString": "mongodb://mongodb/bitmech",
                    "dependencies": [
                            {
                                    "module": "mongojs",
                                    "version": "2.4.0"
                                    }
                            ],
                    "path": "plugins/mongodb",
                    "version": 0.1
                    },
            "paperTrader": {
                    "enabled": True,
                    "feeMaker": 0.15,
                    "feeTaker": 0.25,
                    "feeUsing": "maker",
                    "reportInCurrency": True,
                    "simulationBalance": {
                            "asset": 1,
                            "currency": 100
                            },
                    "slippage": 0.05
                    },
            "performanceAnalyzer": {
                    "enabled": True,
                    "riskFreeReturn": 5
                    },
            "postgresql": {
                    "connectionString": "postgres://user:pass@localhost:5432",
                    "database": None,
                    "dependencies": [
                            {
                                    "module": "pg",
                                    "version": "6.1.0"
                                    }
                            ],
                    "path": "plugins/postgresql",
                    "schema": "public",
                    "version": 0.1
                    },
            "pushbullet": {
                    "email": "jon_snow@westeros.org",
                    "enabled": False,
                    "key": "xxx",
                    "muteSoft": True,
                    "sendMessageOnStart": True,
                    "tag": "[Bitmech]"
                    },
            "pushover": {
                    "enabled": False,
                    "key": "",
                    "muteSoft": True,
                    "sendPushoverOnStart": False,
                    "tag": "[Bitmech]",
                    "user": ""
                    },
            "redisBeacon": {
                    "broadcast": [
                            "candle"
                            ],
                    "channelPrefix": "",
                    "enabled": False,
                    "host": "127.0.0.1",
                    "port": 6379
                    },
            "slack": {
                    "channel": "",
                    "enabled": False,
                    "muteSoft": True,
                    "sendMessageOnStart": True,
                    "token": ""
                    },
            "sqlite": {
                    "dataDirectory": "history",
                    "dependencies": [],
                    "path": "plugins/sqlite",
                    "version": 0.1
                    },
            "talib-macd": {
                    "parameters": {
                            "optInFastPeriod": 10,
                            "optInSignalPeriod": 9,
                            "optInSlowPeriod": 21
                            },
                    "thresholds": {
                            "down": -0.025,
                            "up": 0.025
                            }
                    },
            "telegrambot": {
                    "botName": "gekkobot",
                    "emitUpdates": False,
                    "enabled": False,
                    "token": "YOUR_TELEGRAM_BOT_TOKEN"
                    },
            "trader": {
                    "enabled": False,
                    "key": "",
                    "passphrase": "",
                    "secret": "",
                    "username": ""
                    },
            "tradingAdvisor": {
                    "adapter": "sqlite",
                    "candleSize": 1,
                    "enabled": True,
                    "historySize": 3,
                    "method": "MACD",
                    "talib": {
                            "enabled": False,
                            "version": "1.0.2"
                            }
                    },
            "twitter": {
                    "access_token_key": "",
                    "access_token_secret": "",
                    "consumer_key": "",
                    "consumer_secret": "",
                    "enabled": False,
                    "muteSoft": False,
                    "sendMessageOnStart": False,
                    "tag": "[Bitmech]"
                    },
            "varPPO": {
                    "momentum": "TSI",
                    "thresholds": {
                            "persistence": 0,
                            "weightHigh": -120,
                            "weightLow": 120
                            }
                    },
            "watch": {
                    "asset": "BTC",
                    "currency": "USDT",
                    "exchange": "poloniex"
                    },
            "xmppbot": {
                    "client_host": "jabber_server",
                    "client_id": "jabber_id",
                    "client_port": 5222,
                    "client_pwd": "jabber_pw",
                    "emitUpdates": False,
                    "enabled": False,
                    "receiver": "jabber_id_for_updates",
                    "status_msg": "I'm online"
                    },
            'hyperopt': {'deltaDays': 21,
                        'testDays': 90,
                        'num_rounds': 10,
                        'random_state': 2017,
                        'num_iter': 10,
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
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "short": hp.uniform("short", 0.1,30), # short EMA
                               "longWeight": hp.uniform("longWeight", 1.,5.), # long EMA(short*longWeight)
                               "thresholds.down": hp.uniform("thresholds.down", -0.5,0.), # trend thresholds
                               "thresholds.up": hp.uniform("thresholds.up", 0.,0.5), # trend thresholds
                        },
                        "MACD":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "short": hp.uniform("short", 0.1,30), # short EMA
                               "longWeight": hp.uniform("longWeight", 1.,5.), # long EMA(short*longWeight)
                               "signal": hp.uniform("signal", 1,18), # shortEMA - longEMA diff
                               "thresholds.down": hp.uniform("thresholds.down", -5.,0.), # trend thresholds
                               "thresholds.up": hp.uniform("thresholds.up", 0.,5.), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "PPO":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "short": hp.uniform("short", 0.1,30), # short EMA
                               "longWeight": hp.uniform("longWeight", 1.,5.), # long EMA(short*longWeight)
                               "signal": hp.uniform("signal", 1,18), # 100 * (shortEMA - longEMA / longEMA)
                               "thresholds.down": hp.uniform("thresholds.down", -5.,0.), # trend thresholds
                               "thresholds.up": hp.uniform("thresholds.up", 0.,5.), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        # Uses one of the momentum indicators but adjusts the thresholds when PPO is bullish or bearish
                        # Uses settings from the ppo and momentum indicator config block
                        "varPPO":{ # TODO: merge PPO config
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "short": hp.uniform("short", 0.1,30), # short EMA
                               "longWeight": hp.uniform("longWeight", 1.,5.), # long EMA(short*longWeight)
                               "signal": hp.uniform("signal", 1,18), # 100 * (shortEMA - longEMA / longEMA)
                               "thresholds.down": hp.uniform("thresholds.down", -5.,0.), # trend thresholds
                               "thresholds.up": hp.uniform("thresholds.up", 0.,5.), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                               "momentum": hp.uniform("momentum", 0, 2.99999), # index of ["RSI", "TSI", "UO"]
                               # new threshold is default threshold + PPOhist * PPOweight
                               "weightLow": hp.uniform("weightLow", 60, 180),
                               "weightHigh": hp.uniform("weightHigh", -60, -180),
                        },
                        "RSI":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "interval": hp.uniform("interval", 7,21), # weight
                               "thresholds.low": hp.uniform("thresholds.low", 0,50), # trend thresholds
                               "thresholds.high": hp.uniform("thresholds.high", 50.,100), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "StochRSI":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "interval": hp.uniform("interval", 7,21), # weight
                               "thresholds.low": hp.uniform("thresholds.low", 0,50), # trend thresholds
                               "thresholds.high": hp.uniform("thresholds.high", 50.,100), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "CCI":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "consistant": hp.uniform("consistant", 7,21), # constant multiplier. 0.015 gets to around 70% fit
                               "history": hp.uniform("history", 45,135), # history size, make same or smaller than history
                               "thresholds.down": hp.uniform("thresholds.down", -50,-150), # trend thresholds
                               "thresholds.up": hp.uniform("thresholds.up", 50,150), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                        },
                        "TSI":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "short": hp.uniform("short", 0.1,30), # short EMA
                               "longWeight": hp.uniform("longWeight", 1.,5.), # long EMA(short*longWeight)
                               "thresholds.low": hp.uniform("thresholds.low", -13,-52), # trend thresholds
                               "thresholds.high": hp.uniform("thresholds.high", 13,52), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                         },
                        "gannswing":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "vixperiod": hp.uniform("vixperiod", 1,80), # 
                               "swingperiod": hp.uniform("swingperiod", 1,40), # 
                               "stoploss.enabled": hp.uniform("stoploss.enabled", 0,1), # 
                               "stoploss.trailing": hp.uniform("stoploss.trailing", 0,1), # 
                               "stoploss.percent": hp.uniform("stoploss.percent", 0,5), # 
                         },
                        "UO":{
                               "candleSize": hp.uniform("candleSize", 1,60), # tick per day
                               "historySize": hp.uniform("historySize", 1,60),
                               "first.weight": hp.uniform("first.weight", 2,8), # 
                               "first.period": hp.uniform("first.period", 4.5,14), # 
                               "second.weight": hp.uniform("second.weight", 1,4), # 
                               "second.period": hp.uniform("second.period", 7,28), # 
                               "third.weight": hp.uniform("third.weight", 0.5,2), # 
                               "third.period": hp.uniform("third.period", 14,56), # 
                               "thresholds.low": hp.uniform("thresholds.low", 15,45), # trend thresholds
                               "thresholds.high": hp.uniform("thresholds.high", 45,140), # trend thresholds
                               "thresholds.persistence": hp.uniform("thresholds.persistence", 0,100), # trend duration(count up by tick) thresholds
                         },
            },
        }
