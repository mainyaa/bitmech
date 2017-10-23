#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trollius
import datetime
import pandas as pd
import numpy as np
import argparse
import tqdm
import math
from retry import retry
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
from pubnub.callbacks import SubscribeCallback
from pubnub.enums import PNOperationType, PNStatusCategory

import config
import exchange

from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)
dict_merge = lambda a,b: a.update(b) or a
import inspect

def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr

class RealtimeExchange(object):
    def __init__(self):
        self.setting = config.get_setting()
        self.result = []
        self.exchangename = self.setting["realtime"]["exchange"]
        self.pair = self.setting["realtime"]["pair"]
        self.currency = self.pair.split("_")[0]
        self.asset = self.pair.split("_")[1]
        self.strategy = self.setting["realtime"]["strategy"]
        self.exchange = getattr(exchange, self.exchangename)(self.setting)

    def init_board(self):
        board = self.exchange.board()
        bids = pd.DataFrame.from_dict(board["bids"])
        asks = pd.DataFrame.from_dict(board["asks"])
        bids["type"] = "bids"
        asks["type"] = "asks"
        self.board = pd.concat([bids.iloc[:20,:], asks.iloc[:20,:]])
        self.board.index = self.board["price"]
        board['date'] = datetime.datetime.now()
        self.board = self.board.sort_values("price")
        self.ticker = None
        self.executions = None

    # TODO: replace RxPy
    @retry(Exception, tries=30)
    def watch(self, message_callbacks=None, status_callbacks=None):
        self.init_board()
        pnconfig = PNConfiguration()
        pnconfig.subscribe_key = "sub-c-52a9ab50-291b-11e5-baaa-0619f8945a4f"
        pnconfig.publish_key = ""
        pnconfig.ssl = False
        if message_callbacks is None:
            message_callbacks = {
                    'lightning_board_BTC_JPY': self.lightning_board_BTC_JPY,
                    'lightning_ticker_BTC_JPY': self.lightning_ticker_BTC_JPY,
                    'lightning_executions_BTC_JPY': self.lightning_executions_BTC_JPY,
                    }
        def noop(a,b):
            pass
        if status_callbacks is None:
            status_callbacks = {}
            status_callbacks[PNStatusCategory.PNConnectedCategory] = noop
            status_callbacks[PNStatusCategory.PNReconnectedCategory] = noop
            status_callbacks[PNStatusCategory.PNDisconnectedCategory] = noop
            status_callbacks[PNStatusCategory.PNUnexpectedDisconnectCategory] = noop
            status_callbacks[PNStatusCategory.PNAccessDeniedCategory] = noop

        pubnub = PubNub(pnconfig)
        pubnub.subscribe().channels(message_callbacks.keys()).execute()
        

        class BitflyerSubscribe(SubscribeCallback):
            def __init__(self, realtimeexchange, message_callbacks, status_callbacks):
                self.realtimeexchange = realtimeexchange
                self.message_callbacks = message_callbacks
                self.status_callbacks = status_callbacks
            def status(self, pubnub, status):
                pubnub = props(pubnub)
                status = props(status)
                if status["operation"] == PNOperationType.PNSubscribeOperation \
                        or status["operation"] == PNOperationType.PNUnsubscribeOperation:
                    if status["category"] in self.status_callbacks:
                        self.status_callbacks[status["category"]](pubnub, status)
                    else:
                        pass
                elif status.operation == PNOperationType.PNSubscribeOperation:
                    if status.is_error():
                        pass
                    else:
                        pass
                else:
                    pass

            def presence(self, pubnub, presence):
                pass

            def message(self, pubnub, message):
                pubnub = props(pubnub)
                message = props(message)
                ch = message["channel"]
                if ch in self.message_callbacks:
                    self.message_callbacks[ch](pubnub, message)

        bs = BitflyerSubscribe(self, message_callbacks, status_callbacks)
        pubnub.add_listener(bs)
        self.start_csv_loop()

    def timetoken2datetime(self, timetoken):
        # https://gist.github.com/scalabl3/c2bb54301e0c34f89d7e
        return pd.to_datetime(timetoken / 10000, unit="ms")

    def lightning_executions_BTC_JPY(self, pubnub, message):
        #{'publisher': u'53e0ea18-ac3f-4397-bffa-dbf9b6bb03e2', 'timetoken': 15085215449113979, 'user_metadata': None, 'message': [{u'price': 673900.0, u'exec_date': u'2017-10-20T17:45:44.6738083Z', u'side': u'SELL', u'id': 59618812, u'sell_child_order_acceptance_id': u'JRF20171020-174544-170957', u'buy_child_order_acceptance_id': u'JRF20171020-174544-170956', u'size': 0.19147944}, {u'price': 673856.0, u'exec_date': u'2017-10-20T17:45:44.6738083Z', u'side': u'SELL', u'id': 59618813, u'sell_child_order_acceptance_id': u'JRF20171020-174544-170957', u'buy_child_order_acceptance_id': u'JRF20171020-174543-291374', u'size': 1.37675211}], 'channel': u'lightning_executions_BTC_JPY', 'subscription': None}
        executions = pd.DataFrame.from_dict(message["message"])
        executions["timetoken"] = message["timetoken"]
        executions['exec_date'] = pd.to_datetime(executions['exec_date'])
        executions.index = executions["exec_date"]
        if self.executions is None:
            self.executions = executions
        else:
            self.executions = pd.concat([self.executions, executions])
    def lightning_ticker_BTC_JPY(self, pubnub, message):
        #'publisher': u'53e0ea18-ac3f-4397-bffa-dbf9b6bb03e2', 'timetoken': 15085215449278656, 'user_metadata': None, 'message': {u'volume_by_product': 21842.65357382, u'best_bid': 673856.0, u'best_bid_size': 0.62324789, u'timestamp': u'2017-10-20T17:45:44.6738083Z', u'total_ask_depth': 1280.27643124, u'best_ask': 673904.0, u'tick_id': 24380394, u'volume': 21842.65357382, u'ltp': 673856.0, u'best_ask_size': 0.1, u'total_bid_depth': 5070.91630324, u'product_code': u'BTC_JPY'}, 'channel': u'lightning_ticker_BTC_JPY', 'subscription': None}
        ticker = pd.DataFrame.from_dict([message["message"]])
        ticker["timetoken"] = message["timetoken"]
        ticker['date'] = self.timetoken2datetime(ticker['timetoken'])
        ticker.index = ticker["date"]
        if self.ticker is None:
            self.ticker = ticker
        else:
            self.ticker = pd.concat([self.ticker, ticker])

    def lightning_board_BTC_JPY(self, pubnub, message):
        # {'publisher': u'53e0ea18-ac3f-4397-bffa-dbf9b6bb03e2', 'timetoken': 15085198896631891, 'user_metadata': None, 'message': {u'mid_price': 673408.0, u'bids': [{u'price': 652277.0, u'size': 0.0}], u'asks': []}, 'channel': u'lightning_board_BTC_JPY', 'subscription': None}
        board = message["message"]
        bids = pd.DataFrame.from_dict(board["bids"])
        asks = pd.DataFrame.from_dict(board["asks"])
        bids["type"] = "bids"
        asks["type"] = "asks"
        board = pd.concat([bids, asks])
        board["timetoken"] = message["timetoken"]
        board['date'] = self.timetoken2datetime(board['timetoken'])
        board.index = board["price"]
        board = board.sort_values("price")
        if self.board is None:
            self.board = board
        else:
            delprice = board["price"][board["size"] == 0]
            newboard = board[board["size"] != 0]
            self.board.drop(self.board["price"][self.board["price"].isin(delprice)])
            self.board = pd.concat([self.board, newboard])
            self.board = self.board.sort_values("price")
    
    def start_csv_loop(self):
        interval = 1
        print("save csv each {} seconds".format(interval))
        self.loop = trollius.get_event_loop()
        self.loop.call_soon(self.loop_output_csv, interval)
        self.loop.run_forever()
        self.loop.close()

    def loop_output_csv(self, interval):
        if self.board is not None:
            filename = './data/realtimeexchange/result.realtimeexchange.{}.{}.{}.{}.csv'.format(self.exchangename, self.pair, "board", datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.board.to_csv(filename, index=None)
            #print(filename)
        if self.ticker is not None:
            filename = './data/realtimeexchange/result.realtimeexchange.{}.{}.{}.csv'.format(self.exchangename, self.pair, "ticker")
            self.ticker.to_csv(filename, index=None)
            #print(filename)
        if self.executions is not None:
            filename = './data/realtimeexchange/result.realtimeexchange.{}.{}.{}.csv'.format(self.exchangename, self.pair, "executions")
            self.executions.to_csv(filename, index=None)
            #print(filename)
        self.loop.call_later(interval, self.loop_output_csv, interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['watch'])
    args = parser.parse_args()
    if args.mode == "watch":
        realtime = RealtimeExchange()
        realtime.watch()
    else:
        parser.print_help()
