import numpy as np


def solve_knapsack(v, s, b):
    def f(b):
        if b == 0:
            return 0, -1
        if b < 0:
            return -999999, -1
        if b in memo:
            return memo[b]
        else:
            max_value = 0
            prev = -1
            for i, size in enumerate(s):
                if f(b - size)[0] + v[i] > max_value:
                    max_value = f(b - size)[0] + v[i]
                    prev = i
            memo[b] = max_value, prev
        return memo[b]

    memo = {}
    opt_val, prev = f(b)
    x = [0 for i in range(len(s))]
    while True:
        val, prev = memo[b]
        x[prev] += 1
        b -= s[prev]
        if b <= 0:
            break

    return opt_val, x


def knapsack(n_vars: int):
    v = np.ones(n_vars) + 1
    v[-1] = v[-1] + 2
    w = np.ones(n_vars)
    w_max = 9
    return v, w, w_max


if __name__ == "__main__":
    n_vars = 15
    v, w, w_max = knapsack(n_vars)

    opt_val, x = solve_knapsack(w, v, w_max)
    print("Opt. value=", opt_val)
    print("Sol.=", x)
