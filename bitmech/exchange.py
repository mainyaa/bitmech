#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pybitflyer
import pandas as pd
import datetime
from retry import retry

class Bitflyer(object):
    def __init__(self, setting):
        self.setting = setting
        self.exchangename = self.setting["realtime"]["exchange"]
        self.pair = self.setting["realtime"]["pair"]
        self.api_key = self.setting["bitflyer"]["api_key"]
        self.api_secret = self.setting["bitflyer"]["api_secret"]
        self.api = pybitflyer.API(api_key=self.api_key, api_secret=self.api_secret)
        self.pirivatethreshold = 200 # per minutes per key
        self.publicthreshold = 500 # per minutes per ip
        self.expireorder = 30 # minute max:43200 (30 days)
        self.minimalOrder = 0.00000001 # 1 satoshi

    def getbalance(self):
        balancelist = self.api.getbalance()
        return {x["currency_code"]:x for x in balancelist}

    def board(self):
        return self.api.board(product_code=self.pair)

    def ticker(self):
        return self.api.ticker(product_code=self.pair)

    def buy(self, price, amount):
        return self.send_childorder("BUY", price, amount)

    def sell(self, price, amount):
        return self.send_childorder("SELL", price, amount)

    def send_childorder(self, side, price, amount):
        """
        product_code: Required. The product being ordered. Please specify a product_code or alias, as obtained from the Market List. Only "BTC_USD" is available for U.S. accounts.
        child_order_type: Required. For limit orders, it will be "LIMIT". For market orders, "MARKET".
        side: Required. For buy orders, "BUY". For sell orders, "SELL".
        price: Specify the price. This is a required value if child_order_type has been set to "LIMIT".
        size: Required. Specify the order quantity.
        minute_to_expire: Specify the time in minutes until the expiration time. If omitted, the value will be 43200 (30 days).
        time_in_force: Specify any of the following execution conditions - "GTC", "IOC", or "FOK". If omitted, the value defaults to "GTC".
        """
        params = {
                        "product_code": self.pair,
                        "child_order_type": "LIMIT",
                        "side": side,
                        "price": price,
                        "size": amount,
                        "minute_to_expire": self.expireorder,
                        "time_in_force": "GTC"
                        }
        return self.api.sendchildorder(**params)["child_order_acceptance_id"]

    def get_fee(self):
        params = {
                "product_code": self.pair,
                }
        return self.api.gettradingcommission(**params)["commission_rate"] * 100

    def cancel_order(self, order):
        params = {
                "product_code": self.pair,
                "child_order_id": order,
                }
        return self.api.cancelchildorder(**params)

    def check_order(self, order):
        params = {
                "product_code": self.pair,
                "child_order_id": order,
                }
        return self.api.getchildorders(**params)[0]["child_order_state"]

    @retry(Exception, tries=30)
    def get_trade(self, from_date, count=500):
        executions = self.api.executions(product_code=self.pair, count=count)
        executions = pd.DataFrame.from_dict(executions)
        executions["exec_date"] = pd.to_datetime(executions["exec_date"])
        executions.index = executions["exec_date"]
        # get bitflyer for target date(today - 90)
        end_date = executions.iloc[-1,1]
        while from_date < end_date:
            end_date = executions.iloc[-1,1]
            before = executions.iloc[-1,2]
            e1 = self.api.executions(product_code="BTC_JPY", count=500, before=before)
            e1 = pd.DataFrame.from_dict(e1)
            e1["exec_date"] = pd.to_datetime(e1["exec_date"])
            e1.index = e1["exec_date"]
            executions = executions.append(e1)
        executions.to_csv("data/realtime_bitflyer_%s_executions.csv"%self.pair, index=None)
        return executions

def bitflyer(setting):
    return Bitflyer(setting)

