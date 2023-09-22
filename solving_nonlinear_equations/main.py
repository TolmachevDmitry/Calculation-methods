import matplotlib.pyplot as plt
import numpy as np
import algorithms as alg


def f(x):
    return 0.5 * np.exp(-x ** 2) + x * np.cos(x)


def df(x):
    return -x * np.exp(-x ** 2) - x * np.sin(x)


x = np.linspace(0, 10, 100)

intervals, x0_for_newton, titles = [[0, 2], [4, 6], [6, 8]], [1, 4, 7], ["Bisection", "Chord", "Newton"]
functions = [alg.bisection, alg.chord, alg.newton]

root_list, iter_list = list(), list()
for fu in functions:
    print(f"{titles[len(root_list)]}:")
    root_list_0, iter_list_0 = list(), list()
    for val in intervals if fu != alg.newton else x0_for_newton:
        root, count_iter = fu(f, val[0], val[1]) if fu != alg.newton else fu(f, df, val)
        root_list_0.append(root)
        iter_list_0.append(count_iter)

        print(f"x{len(root_list_0)}: {root}")
        print(f"Количество итераций: {count_iter}")
    root_list.append(root_list_0)
    iter_list.append(iter_list_0)
    print("")


for i in range(3):
    plt.title(titles[i])
    plt.plot(x, f(x))
    for j in root_list[i]:
        plt.scatter(j, f(j), c='r')
    plt.grid(True)
    plt.show()

# a, it = alg.newton(f, df, 4)
# print(a)
#
# plt.title("")
# plt.plot(x, f(x))
# plt.scatter(a, f(a), c='r')
# plt.grid(True)
# plt.show()
