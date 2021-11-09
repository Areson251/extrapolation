import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt
import random


def exponential_fit(x, a, b, c):
    return a * np.exp(-b * x) + c


if __name__ == "__main__":
    prices = []

    # kol_of_rand = 5000
    # for i in range(kol_of_rand):
    #     prices.append(random.randint(100, 200))
    # prices.append(random.randint(100, 200))

    file = open('prices.txt', 'r')
    for line in file:
        prices.append(float(line))

    l = len(prices)
    kol_of_prediction = 900
    x = np.arange(0, l, 1)
    y = np.array(prices)

    plt.plot(x, y, color="blue")
    # plt.scatter(x, y, color="blue")

    # plt.plot(y)
    # plt.show()

    kol_of_rand = l
    x1 = np.arange(kol_of_rand, kol_of_rand+kol_of_prediction, 1)

    # x1 = np.array([l-1])
    # y1 = np.array(y[-1])
    x1 = [l-1]
    y1 = [y[-1]]


    f = interpolate.interp1d(x, y, fill_value="extrapolate")
    for i in range(kol_of_prediction):
        print(f(l+i))
        # np.append(y, f(l+i))
        # np.append(x, l+i)
        y1.append(f(l+i))
        x1.append(l+i)
    #
    plt.plot(x1, y1, color="red")
    plt.show()

    # for k in (1, 2, 3):
    #     extrapolator = interpolate.UnivariateSpline(x, y, k=k)
    #     y1 = extrapolator(x1)
    #     label = "k=%d" % k
    #     plt.plot(x1, y1, label=label)
    extrapolator = interpolate.UnivariateSpline(x, y, k=1)
    y1 = extrapolator(x1)
    label = "k=%d" % 1
    plt.plot(x1, y1, label=label)
    plt.legend(loc="lower left")
    plt.show()

    # fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
    # a, b, c = fitting_parameters
    #
    # next_x = len(prices)
    # next_y = exponential_fit(next_x, a, b, c)

    # plt.plot(y)
    # plt.scatter(next_x, next_y, color="red")
    # plt.plot(np.append(y, next_y), 'ro')
    # plt.show()