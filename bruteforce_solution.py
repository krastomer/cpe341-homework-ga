import time


def formula(x: int):
    return (x ** 3) - (60 * (x ** 2)) + (900 * x) + 150


ans = []

if __name__ == "__main__":
    start = time.time()
    for i in range(1, 64 + 1):
        ans.append(formula(i))

    stop = time.time() - start

    print('min:\t', min(ans))
    print('x:\t', ans.index(min(ans)) + 1)

    print('max:\t', max(ans))
    print('x:\t', ans.index(max(ans)) + 1)

    print('{:.5}'.format(stop))
