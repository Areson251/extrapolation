import numpy as np
from scipy import interpolate

x = [1, 2, 3, 4, 5]
y = [5, 10, 15, 20, 25]

f = interpolate.interp1d(x, y, fill_value="extrapolate")
for i in range(10):
    print(f(len(y)+i))