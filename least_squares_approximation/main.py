import numpy as np
import matplotlib.pyplot as plt


def polyfit(x, y, deg):
    a, b = np.repeat(x, (deg + 1) ** 2).reshape(n, (deg + 1) ** 2).T ** (
            np.repeat(np.arange(0, deg + 1), (deg + 1)).reshape(deg + 1, deg + 1).T + np.arange(0, deg + 1).reshape(
        deg + 1, 1)).reshape((deg + 1) ** 2, 1), np.repeat(y, (deg + 1)).reshape(n, (deg + 1)).T
    return np.array(list(reversed(np.linalg.solve(np.sum(a, axis=1).reshape(deg + 1, deg + 1), np.sum(b * a[0: deg + 1], axis=1).reshape(deg + 1, 1)).T[0])))


left, right, n = 0, 10, 20
x = np.random.uniform(left, right, n)
p_true = np.array([1., 2., 3.])
y_true = np.polyval(p_true, x)
y = y_true + np.random.normal(0, np.abs(y_true).mean() * 0.1, n)
p = polyfit(x, y, 2)
assert np.allclose(p, np.polyfit(x, y, 2))
xx = np.linspace(x.min(), x.max(), 100)
plt.plot(x, y, 'o', xx, np.polyval(p_true, xx), '--', xx, np.polyval(p, xx))
plt.grid(True)
plt.show()
