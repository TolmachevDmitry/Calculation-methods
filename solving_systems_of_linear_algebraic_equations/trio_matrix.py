import numpy as np
from random import uniform as rand


# Создаие трёхдиагональной матрицы
def trio(n):
    ind = [1 if i % (n - 1) == 0 else 0 for i in range((n - 2) * (n - 1) + n)]
    val = np.array(range(0, n * n), float) * 0
    
    

    i = k = 0
    while k < len(ind):
        d = 0
        for j in range(3 * ind[k] - (1 if k == 0 or k == len(ind) - 1 else 0)):
            val[i + j] = rand(1, 100)
            d = j

        k, i = k + 1, i + 1 + d

    return val.reshape(n, n)


