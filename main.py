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
    oc_dif, hl_dif = 0, 0

    for candle in last_can:
        if (candle.open - candle.close) > oc_dif: oc_dif = candle.open - candle.close
        if (candle.high - candle.low) > hl_dif: hl_dif = candle.high - candle.low

    for price in prices:
        open = random.uniform(price - oc_dif/2, price + oc_dif/2)
        high = random.uniform(price - hl_dif/2, price + hl_dif/2)
        low = random.uniform(price - hl_dif/2, price + hl_dif/2)
        candle = Candle(open, price, high, low, times[i], resolution)
        candles.append(candle)
        i += 1

    return candles


def linear_extrapolation(candles, times):
    cnt = len(times)
    prices = [candle.close for candle in candles]
    time = [x for x in range(len(candles))]

    l = len(prices)
    kol_of_prediction = cnt
    x = pd.DataFrame(np.array(time))
    y = pd.DataFrame(np.array(prices))

    kol_of_rand = l
    x1 = np.arange(kol_of_rand, kol_of_rand + kol_of_prediction, 1)

    extrapolator = interpolate.UnivariateSpline(x, y, k=1)
    y1 = extrapolator(x1)
    # print(y1.tolist())
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


def interpolation(candles, times):
    cnt = len(times)
    prices = [candle.close for candle in candles]
    k = len(prices)
    x = [i for i in range(k)]
    data = []

    f = interpolate.interp1d(x, prices, fill_value="extrapolate")
    for i in range(cnt):
        data.append(f(k + i))
    res = generate_candles(candles, data, times)
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
    x, ya = [], []
    for line in file_candles:
        l = line[:-1].split(' ')
        a = datetime.datetime.fromtimestamp(float(l[4]))
        candle = Candle(float(l[0]), float(l[1]), float(l[2]), float(l[3]), a, l[5])
        candles.append(candle)
        x.append(float(l[4]))
        ya.append(float(l[1]))

    x1 = []
    file_times = open('times.txt', 'r')
    for line in file_times:
        times.append(datetime.datetime.fromtimestamp(float(line)))
        x1.append(float(line))

    fig = plt.figure()
    # plt.plot(x, ya, color="blue")

    y1 = linear_extrapolation(candles, times)

    y2 = linear_regression(candles, times)

    y3 = interpolation(candles, times)
