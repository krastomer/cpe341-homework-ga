def formula(x: int):
    return (x ** 3) - (60 * (x ** 2)) + (900 * x) + 150


ans = []

if __name__ == "__main__":
    for i in range(1, 64 + 1):
        ans.append(formula(i))

    print('min:\t', min(ans))
    print('x:\t', ans.index(min(ans)) + 1)

    print('max:\t', max(ans))
    print('x:\t', ans.index(max(ans)) + 1)
