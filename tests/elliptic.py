import numpy as np
from scipy.special import ellipe, ellipk, ellipkm1
import matplotlib.pyplot as plt

import traceon.backend as backend
import traceon.util as util

print(ellipkm1(1e-20))
print(backend.ellipkm1(1e-20))

# ellipkm1

plt.figure()
x = np.linspace(1, 100, 20)
plt.plot(x, backend.ellipkm1(10**(-x))/ellipkm1(10**(-x)) - 1)
#plt.plot(x, backend.ellipkm1(10**(-x)), color='black', linestyle='dashed')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('ellipk(1 - 10^-x)')

# ellipem1

plt.figure()
x = np.linspace(1, 100, 20)
plt.plot(x, backend.ellipem1(10**(-x))/ellipe(1 - 10**(-x)) - 1)
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('ellipe(1 - 10^-x)')


## Complete elliptic integral first kind

x = np.linspace(-100, 10, 500)

plt.figure()
plt.title('Ellipk')
plt.plot(x, ellipk(x))
K = np.array([backend.ellipk(x_) for x_ in x] )
plt.plot(x, K, linestyle='dashed')

plt.figure()
plt.title('Ellipe')
plt.plot(x, ellipe(x))
E = np.array([backend.ellipe(x_) for x_ in x])
plt.plot(x, E, linestyle='dashed')

plt.figure()
plt.plot(x, K/ellipk(x) - 1, label='Ellipk')
plt.plot(x, E/ellipe(x) - 1, label='Ellipe')
plt.legend()
plt.yscale('log')

plt.show()
