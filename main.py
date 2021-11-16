import numpy as np
from scipy import interpolate
from sklearn import linear_model
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

def linear_extrapolation(prices, cnt):
    l = len(prices)
    kol_of_prediction = cnt
    x = pd.DataFrame(np.arange(0, l, 1))
    y = pd.DataFrame(np.array(prices))

    kol_of_rand = l
    x1 = np.arange(kol_of_rand, kol_of_rand + kol_of_prediction, 1)

    # x1 = [l - 1]
    # y1 = [y[-1]]

    # lm = linear_model.LinearRegression()
    # model = lm.fit(x, y)
    #
    # predictions = lm.predict(x)
    #
    # f = interpolate.interp1d(x, y, fill_value="extrapolate")
    # for i in range(kol_of_prediction):
    #     y1.append(f(l + i))
    #     x1.append(l + i)

    extrapolator = interpolate.UnivariateSpline(x, y, k=1)
    y1 = extrapolator(x1)
    return y1.tolist()


def linear_regression(prices, cnt):
    data=[]
    for i in range(cnt):
        model = AutoReg(prices, lags=1)
        model_fit = model.fit()
        yhat = model_fit.predict(len(prices), len(prices))
        prices.append(yhat.tolist()[0])
        data.append(yhat.tolist()[0])
    return data


if __name__ == "__main__":
    prices, x, i, cnt = [], [], 0, 900
    file = open('prices.txt', 'r')
    for line in file:
        prices.append(float(line))
        x.append(i)
        i+=1

    fig = plt.figure()
    plt.plot(x, prices, color="blue")
    x.clear()
    while i <(len(prices)+cnt):
        x.append(i)
        i+=1

    y1 = linear_extrapolation(prices, cnt)
    plt.plot(x, y1, color="red")
    y2 = linear_regression(prices, cnt)
    # plt.plot(x, y2, color="green")
    # plt.show()
