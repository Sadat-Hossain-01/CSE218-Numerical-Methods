from matplotlib import pyplot as pt
import numpy as np


# The equation is x^3 - 0.18 * (x^2) + 4.752e-4, where x is the depth to be found (in meter)


def evaluate(x):
    return x ** 3 - 0.18 * (x ** 2) + 4.752e-4


def plot():
    xpoints = np.arange(-1, 1, step=0.001)
    pt.plot(xpoints, evaluate(xpoints))
    pt.grid(True, which='both')
    pt.axhline(y=0, color='g')
    pt.xlabel("x (meter)", fontdict={'fontname': 'Comic Sans MS'})
    pt.ylabel("f(x)", fontdict={'fontname': 'Comic Sans MS'})
    pt.title("Graph for Visual Representation", fontdict={
             'fontname': 'Comic Sans MS', 'fontsize': 20})
    pt.plot(0, evaluate(0), 'co')
    pt.plot(0.12, evaluate(0.12), 'co')
    pt.show()


def bisection(low, high, error_limit, max_iteration):
    step = 1
    mid = (low + high) / 2
    error = None

    while True:
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

        step += 1


def show_table(low, high, error_limit, max_iteration):
    step = 1
    mid = (low + high) / 2
    error = None
    space = "            "
    print("Step    " + space + "Low       " + space + "High      " +
          space + "Middle    " + space + "Error")

    while step <= max_iteration:
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
            print(format(error, ".8f"))
        else:
            print("N/A")

        step += 1


plot()

# After seeing the plot, we are taking our guesses to be 0m and 0.12m respectively
soln = bisection(0, 0.12, 0.5, 20)

print(
    f'The depth to which the ball is submerged is {format(soln*100, ".6f")} cm')
show_table(0, 0.12, 0.5, 20)
