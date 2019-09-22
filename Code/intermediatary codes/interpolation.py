import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

x = np.arange(0, 10)
y = np.arange(0,10)
f = interpolate.interp1d(x, y, kind='cubic')
f1 = interpolate.interp1d(x, y, kind='quadratic')

xnew = np.arange(0,9, 0.1)
ynew = f(xnew)
# use interpolation function returned by `interp1d`
xnew1 = np.arange(0,9, 0.1)
ynew1 = f1(xnew1) 

plt.plot(x, y, 'o', xnew, ynew, '-')
plt.plot(x, y, 'o', xnew1, ynew1, '--')
plt.show()
