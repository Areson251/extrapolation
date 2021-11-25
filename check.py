# MA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
import matplotlib.pyplot as plt
import copy

# contrived dataset
data = [x + random() for x in range(1, 100)]
data1 = copy.deepcopy(data)
data2 = []
# fit model
for i in range(10):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    data.append(yhat.tolist()[0])
    data2.append(yhat.tolist()[0])
    print(yhat)

x1, x2 = [], []
for i in range(len(data1)):
    x1.append(i)
for i in range(len(data2)):
    x2.append(len(data1)+i-1)

plt.plot(x1, data1)
plt.plot(x2, data2, color="red")
plt.show()
