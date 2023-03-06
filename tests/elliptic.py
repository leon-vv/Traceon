import numpy as np
from scipy.special import ellipe, ellipk
import matplotlib.pyplot as plt

import traceon.backend as backend
import traceon.util as util

## Complete elliptic integral first kind

x = np.linspace(-100, 10, 500)

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
