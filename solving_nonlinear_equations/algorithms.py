import numpy as np


def bisection(f, a, b, xtol=1e-6, ftol=1e-6):
    i = 0

    while True:
        c = (a + b) / 2
        if (np.absolute(f(c)) < ftol) or ((b - a) < xtol):
            break

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        i += 1

    return c, i + 1


def chord(f, a, b, xtol=1e-6, ftol=1e-6):
    i = 0

    while True:
        c = a - f(a) * ((b - a) / (f(b) - f(a)))

        if (np.absolute(f(c)) < ftol) or ((b - a) < xtol):
            break

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        i += 1

    return c, i + 1


def newton(f, df, x0, xtol=1e-50, max_iter=500):
    x_prev, i = x0, 0

    while True:
        x = x_prev - f(x_prev) / df(x_prev)

        if (np.absolute(x - x_prev) < xtol) or (np.absolute(f(x)) < xtol):
            return x, i + 1

        x_prev = x

    print('Превышено максимальное число итераций!!!')
    return x, i + 1


def simple_iteration(g, x0, xtol=1e-6, max_iter=500):
    for i in range(max_iter):
        x = g(x0)
        if np.absolute(x - x0) <= xtol:
            return x, i + 1
        x0 = x

    print('Превышено максимальное число итераций!!!')
    return x, i + 1
