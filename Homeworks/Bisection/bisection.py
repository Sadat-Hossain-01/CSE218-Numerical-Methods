from matplotlib import pyplot as pt
import numpy as np


def evaluate(x):
    return x ** 3 - 0.18 * (x ** 2) + 4.752e-4


def plot():
    xpoints = np.arange(-1, 1, step=0.001)
    pt.plot(xpoints, evaluate(xpoints))
    pt.grid(True, which='both')
    pt.axhline(y=0, color='g')
    pt.xlabel("x")
    pt.ylabel("f(x)")
    pt.title("Graph for Visual Representation")
    pt.plot(0, evaluate(0), 'co')
    pt.plot(0.12, evaluate(0.12), 'co')
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

        if step > max_iteration or (step > 1 and error <= error_limit):
            return mid


def show_table(low, high, error_limit, max_iteration):
    step = 0
    mid = (low + high) / 2
    error = None
    space = "            "
    print("Step    " + space + "Low       " + space + "High      " +
          space + "Middle    " + space + "Error")
    while True:
        step += 1
        old_mid = mid
        mid = (low + high) / 2
        val_l = evaluate(low)
        val_m = evaluate(mid)

        print(format(step, "02d"), end=space + "      ")
        print(format(low, ".8f"), format(high, ".8f"),
              format(mid, ".8f"), sep=space, end=space)

        if val_l * val_m < 0:
            high = mid
        else:
            low = mid

        if step > 1:
            error = abs(((mid - old_mid) / mid) * 100)

        if step > 1:
            print(format(error, ".8f"))
        else:
            print("N/A")

        if step > max_iteration:
            break


plot()
print(bisection(0, 0.12, 0.5, 20))
show_table(0, 0.12, 0.5, 20)
