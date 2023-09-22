import numpy as np
import matplotlib.pyplot as plt


def f1(x1, x2):
    return np.cos(x2 - 1) + x1 - 0.5


def f2(x1, x2):
    return x2 - np.cos(x1) - 3


x1, x2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(1, 5, 100))
plt.contour(x1, x2, f1(x1, x2), 0, colors='blue')
plt.contour(x1, x2, f2(x1, x2), 0, colors='green')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.show()


def f(x):
    return np.array([np.cos(x[1] - 1) + x[0] - 0.5, x[1] - np.cos(x[0]) - 3])


def jac(x):
    return np.array([[1, -np.sin(x[1] - 1)],
                     [np.sin(x[0]), 1]])


def newton(f, jac, x0, tol=1e-6, max_iter=500):
    for i in range(max_iter):
        x = x0 - np.linalg.solve(jac(x0), f(x0))
        if np.linalg.norm(x - x0, np.inf) <= tol:
            return x, i + 1
        x0 = x
    print('Превышено максимальное число итераций.')
    return x, i + 1


x0 = np.array([0., 4.])
x, it = newton(f, jac, x0)
plt.contour(x1, x2, f1(x1, x2), 0, colors='blue')
plt.contour(x1, x2, f2(x1, x2), 0, colors='green')
plt.scatter(x[0], x[1], c='r')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.show()
print(f'Количество итераций равно {it}')


