import numpy as np
import matplotlib.pyplot as plt


def cspline(x, y, xx):
    n, dim = len(x), xx.shape
    h = x[1: n] - x[0: n - 1]

    b = 3 * ((y[2: len(y)] - y[1: len(y) - 1]) / h[1: len(h)] - (y[1: len(y) - 1] - y[0: len(y) - 2]) /
             h[0: len(h) - 1])
    b = np.array([0, *b, 0])

    a = np.zeros((n, n))
    rows, cols = np.indices(a.shape)

    a[rows == cols + 1] = np.array([*h[0: len(h) - 1], 0])
    a[rows == cols - 1] = np.array([0, *h[1: len(h)]])
    a[rows == cols] = np.array([1, *(2 * (h[0: len(h) - 1] + h[1: len(h)])), 1])

    s = np.linalg.solve(a, b)

    xx_t = xx

    p1, p2, p3 = y[:-1], (y[1:] - y[:-1]) / h - h * (s[1:] + 2 * s[:-1]) / 3, (s[1:] - s[:-1]) / (3 * h)
    ind = np.searchsorted(x, xx, side='right') - 1
    ind[ind == n - 1] -= 1

    f = lambda el, num: el[num]

    dx = xx_t - f(x, ind)

    spline = f(p1, ind) + f(p2, ind) * dx + f(s, ind) * dx ** 2 + f(p3, ind) * dx ** 3
    return spline


x = np.array([0.1, 2.2, 3.1, 4.9, 6.5])
y = np.sin(x)
xx = np.linspace(0.1, 6.5, 101)
yy = cspline(x, y, xx)
plt.plot(x, y, 'o', xx, np.sin(xx), '--', xx, yy)
plt.grid()
plt.show()
