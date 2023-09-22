import numpy as np
import matplotlib.pyplot as plt


def explicit_euler(f, y_0, x):
    y = np.zeros(len(x))

    y[0] = y_0
    h = x[1] - x[0]

    for i in range(0, len(y) - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return y


def improved_euler(f, y_0, x):
    y = np.zeros(len(x))
    y_w = np.zeros(len(x))

    y[0] = y_w[0] = y_0
    h = x[1] - x[0]

    for i in range(0, len(y) - 1):
        value_func1 = f(x[i], y[i])
        y_w[i + 1] = y[i] + h * value_func1
        value_func2 = f(x[i + 1], y_w[i + 1])

        y[i + 1] = y[i] + (h / 2) * (value_func1 + value_func2)

    return y


def runge_kutta(f, y_0, x):
    y = np.zeros(len(x))

    y[0] = y_0
    h = x[1] - x[0]

    for i in range(0, len(y) - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2), y[i] + (h / 2) * k1)
        k3 = f(x[i] + (h / 2), y[i] + (h / 2) * k2)
        k4 = f(x[i + 1], y[i] + h * k3)

        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[i + 1] = y[i] + h * k

    return y


def diff_equation(x, y):
    return np.exp(2 * x) * (2 + 3 * np.cos(x)) / (2 * y) - 3 * y * np.cos(x) / 2


def exact_solution_diff_equation(x):
    return np.sqrt(np.exp(2 * x))


a, b = 0, 1.6
count = 20
x = np.linspace(a, b, count)

answers = list()
answers.append(explicit_euler(diff_equation, 1, x))
answers.append(improved_euler(diff_equation, 1, x))
answers.append(runge_kutta(diff_equation, 1, x))

plt.plot(x, answers[0], label=" Explicit Euler's method")
plt.plot(x, answers[1], label="Improve Euler's method")
plt.plot(x, answers[2], label="Fourth-order Runge Kutta method")
plt.plot(x, exact_solution_diff_equation(x), label="Exact solution")
plt.grid(True)
plt.legend()
plt.show()

