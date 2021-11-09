import numpy as np
from scipy import interpolate


def linear_extrapolation(prices, cnt):
    l = len(prices)
    kol_of_prediction = cnt
    x = np.arange(0, l, 1)
    y = np.array(prices)

    kol_of_rand = l
    x1 = np.arange(kol_of_rand, kol_of_rand + kol_of_prediction, 1)

    x1 = [l - 1]
    y1 = [y[-1]]

    f = interpolate.interp1d(x, y, fill_value="extrapolate")
    for i in range(kol_of_prediction):
        y1.append(f(l + i))
        x1.append(l + i)

    extrapolator = interpolate.UnivariateSpline(x, y, k=1)
    y1 = extrapolator(x1)
    return y1.tolist()


if __name__ == "__main__":
    prices = []
    file = open('prices.txt', 'r')
    for line in file:
        prices.append(float(line))

    y1 = linear_extrapolation(prices, 900)
