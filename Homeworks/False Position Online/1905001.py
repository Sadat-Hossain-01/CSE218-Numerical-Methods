import numpy as np
import matplotlib.pyplot as pt
import math


def func(x):
    return x ** 3 - x - 1


def x_r(l, u):
    numerator = u * func(l) - l * func(u)
    denominator = func(l) - func(u)
    return numerator / denominator


def plot():
    xvalues = np.arange(-2, 2, 0.1)
    pt.axhline(y=0)
    pt.axvline(x=0)
    pt.grid(which="both")
    pt.plot(xvalues, func(xvalues), "g")
    pt.title("Plot")
    pt.show()


def solve(lguess, rguess):
    low = lguess
    high = rguess
    old_mid = x_r(low, high)
    mid = old_mid
    step = 1
    error = None

    while True:
        mid = x_r(low, high)

        if func(low) * func(mid) > 0:
            low = mid
        else:
            high = mid

        print("Iteration " + str(step) + ":", end=' ')

        if step > 1 and not mid == 0:
            error = math.fabs((mid - old_mid) / mid) * 100
            print(format(error, ".6f") + "%")
        else:
            print("N/A")

        if step > 1 and error < 0.05:
            return mid

        step += 1
        old_mid = mid


plot()
ans = solve(1, 1.5)
print(f'The only real root for x^3-x-1=0 is {ans}')
