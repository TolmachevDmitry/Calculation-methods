import numpy as np
from trio_matrix import trio


# Метод Зейделя
def seidel(a, b, tol=1e-8, max_iter=500):
    n = len(a)
    c = np.eye(n) - a / np.diag(a)[:, None]
    d = b / np.diag(a)[:, None]
    x, p = d.copy(), d.copy()
    for i in range(max_iter):

        for j in range(n):
            x[j] = c[j] @ x + d[j]

        if np.linalg.norm(x - p, np.inf) <= tol:
            return x, i + 1
        p = np.copy(x)
    print('Превышено максимальное число итераций.')

    return x, i + 1


# Метод прогонки
def sweep_method(a, b):
    n = len(a)

    a_i = (np.array(range(1, n + 1), float) * 0).reshape(n, 1)
    x = b_i = b * 0

    y = a[0][0]
    a_i[0], b_i[0] = -a[0][1] / y, b[0] / y

    for i in range(1, n):
        y = a[i][i] + a[i][i - 1] * a_i[i - 1]

        if i != n - 1:
            a_i[i] = -a[i][i + 1] / y
        b_i[i] = (b[i] - a[i][i - 1] * b_i[i - 1]) / y

    x[n - 1] = b_i[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = a_i[i] * x[i + 1] + b_i[i]

    return x


# Метод Якоби
def jacobi(a, b, tol=1e-8, max_iter=500):
    n = len(a)
    c = np.eye(n) - a / np.diag(a)[:, None]
    d = b / np.diag(a)[:, None]
    x = p = d
    for i in range(max_iter):
        x = c @ x + d
        if np.linalg.norm(x - p, np.inf) <= tol:
            return x, i + 1
        p = x
    print('Превышено максимальное число итераций.')
    return x, i + 1


n = 5
a = np.random.normal(0, 100, (n, n))
for i in range(n):
    a[i, i] = np.sum(np.abs(a[i])) + 1

b = np.random.normal(0, 100, (n, ))

x, it = seidel(a, b)
print(f"count iteration of Seidel's methods: {it}")

x, it = jacobi(a, b)
print(f"count iteration of Jacobi's methods: {it}")
assert np.allclose(a @ x, b)

a = trio(n)
b = np.random.normal(0, 100, (n, 1))

x = sweep_method(a, b)
assert np.allclose(a @ x, b)
