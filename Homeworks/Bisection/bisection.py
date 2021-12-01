import matplotlib.pyplot as pt
import numpy as np


def evaluate(x):
    return x ** 3 - 0.18 * (x ** 2) + 4.752e-4


def plot():
    xpoints = np.arange(-20, 20, step=0.01)
    ypoints = xpoints ** 3 - 0.18 * (xpoints**2) + 4.752e-4
    pt.plot(xpoints, ypoints)
    pt.xlabel("x")
    pt.ylabel("f(x) = x^3 - 0.18 * (x^2) + 0.0004752")
    pt.show()


def bisection(low, high, error_limit, max_iteration):
    step = 0
    mid = (low + high) / 2
    error = None
    while True:
        step += 1
        old_mid = mid
        mid = (low + high) / 2
        val_l = evaluate(low)
        val_m = evaluate(mid)

        if val_l * val_m < 0:
            high = mid
        else:
            low = mid

        if step > 1:
            error = abs(((mid - old_mid) / mid) * 100)

        if step >= max_iteration or (step > 1 and error <= error_limit):
            return mid


def show_table(low, high, error_limit, max_iteration):
    step = 0
    mid = (low + high) / 2
    error = None
    space = "          "
    print("Step    " + space + "Low     " + space + "High    " +
          space + "Middle  " + space + "Error")
    while True:
        step += 1
        old_mid = mid
        mid = (low + high) / 2
        val_l = evaluate(low)
        val_m = evaluate(mid)

        print(format(step, "02d"), end=space + "      ")
        print(format(low, ".6f"), format(high, ".6f"),
              format(mid, ".6f"), sep=space, end=space)

        if val_l * val_m < 0:
            high = mid
        else:
            low = mid

        if step > 1:
            error = abs(((mid - old_mid) / mid) * 100)

        if step > 1:
            print(format(error, ".6f"))
        else:
            print("N/A")

        if step >= max_iteration or (step > 1 and error <= error_limit):
            return


plot()
print(bisection(0, 0.12, 0.5, 20))
show_table(0, 0.12, 0.5, 20)
