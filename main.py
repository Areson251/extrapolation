from scipy import interpolate
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
import numpy as np
import datetime
import random
from math import inf


def generate_candles(last_can, prices, times):
    candles, i, resolution = [], 0, last_can[0].resolution
    min_open, max_open = inf, 0
    min_high, max_high = inf, 0
    min_low, max_low = inf, 0
    for candle in last_can:
        if candle.open < min_open: min_open = candle.open
        if candle.high < min_high: min_high = candle.high
        if candle.low < min_low: min_low = candle.low

        if candle.open > max_open: max_open = candle.open
        if candle.high > max_high: max_high = candle.high
        if candle.low > max_low: max_low = candle.low

    for price in prices:
        open = random.uniform(min_open, max_open)
        high = random.uniform(min_high, max_high)
        low = random.uniform(min_low, max_low)
        candle = Candle(open, price, high, low, times[i], resolution)
        candles.append(candle)
        i += 1

    return candles


def linear_extrapolation(candles, times):
    cnt = len(times)
    prices = [candle.close for candle in candles]
    k = len(prices)

    l = len(prices)
    kol_of_prediction = cnt
    x = pd.DataFrame(np.arange(0, k, 1))
    y = pd.DataFrame(np.array(prices))

    kol_of_rand = l
    x1 = np.arange(kol_of_rand, kol_of_rand + kol_of_prediction, 1)

    extrapolator = interpolate.UnivariateSpline(x, y, k=1)
    y1 = extrapolator(x1)
    res = generate_candles(candles, y1.tolist(), times)
    return res


# Autoregression (AR)
def linear_regression(candles, times):
    cnt = len(times)
    prices = [candle.close for candle in candles]
    k = len(prices)

    model = AutoReg(prices, lags=1)
    model_fit = model.fit()
    yhat = model_fit.predict(k, k+cnt-1)
    res = generate_candles(candles, yhat.tolist(), times)
    return res

@dataclass
class Candle:
    open: float
    close: float
    high: float
    low: float
    date: datetime
    resolution: str


if __name__ == "__main__":
    candles, times = [], []
    file_candles = open('candles.txt', 'r')
    x, y = [], []
    for line in file_candles:
        l = line[:-1].split(' ')
        a = datetime.datetime.fromtimestamp(float(l[4]))
        candle = Candle(float(l[0]), float(l[1]), float(l[2]), float(l[3]), a, l[5])
        candles.append(candle)
        x.append(float(l[4]))
        y.append(float(l[1]))

    x1 = []
    file_times = open('times.txt', 'r')
    for line in file_times:
        times.append(datetime.datetime.fromtimestamp(float(line)))
        x1.append(float(line))

    fig = plt.figure()
    # plt.plot(x, y, color="blue")

    y1 = linear_extrapolation(candles, times)
    # plt.plot(x1, y1, color="red")

    y2 = linear_regression(candles, times)
    # plt.plot(x1, y2, color="red")

    # plt.show()
