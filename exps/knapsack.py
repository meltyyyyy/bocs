

def knapsack(s, v, b):
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


if __name__ == "__main__":
    s = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = 10

    opt_val, x = knapsack(s, v, b)
    print("Opt. value=", opt_val)
    print("Sol.=", x)
