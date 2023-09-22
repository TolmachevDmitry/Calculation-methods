import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def aitken(x, y, xx):
    n, dim = len(y), xx.shape
    xx_t = np.repeat(xx, n).reshape(dim[0], n).T if len(dim) == 1 else np.repeat(xx.flatten(), n).reshape(
        len(xx.flatten()), n).T
    l_i = y.reshape(n, 1)

    for i in range(1, n):
        x_1, x_2 = np.reshape(x[0: n - i], (n - i, 1)), np.reshape(x[i: n], (n - i, 1))
        l_i_n = len(l_i)
        l_i = (l_i[0: l_i_n - 1] * (x_2 - xx_t[0: n - i]) - l_i[1: l_i_n] * (x_1 - xx_t[0: n - i])) * (1 / (x_2 - x_1))

    return l_i[0] if len(dim) == 1 else l_i[0].reshape(dim)


def newton(x, y, xx):
    p_x, prod, n, diffs = y[0], 1, len(y), y

    for i in range(1, len(x)):
        diffs = (diffs[1: len(diffs)] - diffs[0: len(diffs) - 1]) / (x[i: n] - x[0: n - i])

        prod *= (xx - x[i - 1])
        p_x += prod * diffs[0]

    return p_x


x = np.linspace(0, np.pi * 2, 5)
y = np.sin(x)
xx = np.linspace(0, np.pi * 2, 100)
yyl = aitken(x, y, xx)
yyn = newton(x, y, xx)
assert np.allclose(yyl, yyn)
plt.plot(x, y, 'o', xx, np.sin(xx), xx, yyl, '--')
plt.show()
plt.plot(x, y, 'o', xx, np.sin(xx), xx, yyn, '--')
plt.show()


# x = np.linspace(0, np.pi * 2, 5)
# y = np.sin(x)
# xx = np.linspace(0, np.pi * 2, 100)
# xx = np.array([[1, 2], [3, 4]])
# yyl = aitken(x, y, xx)
# yyn = newton(x, y, xx)
# assert np.allclose(yyl, yyn)
# plt.plot(x, y, 'o', xx, np.sin(xx), xx, yyl, '--')
# plt.show()
