from math import exp
import matplotlib.pyplot as pt
import numpy as np

# Bisection Required for Exponential Model
def EquationForBisectionSolve(xvalues, yvalues, current_guess):
    # y = ae^(bx)
    # We are solving for b
    # a = sum_y_ebx / sum_e2bx
    # sum_x_y_ebx - a * sum_x_e2bx = 0, first solve for b, then you will get a (1)
    
    ret = 0
    numerator_a, denominator_a, mult = 0, 0, 0
    n = xvalues.size
    for i in range(n):
        ret = ret + xvalues[i] * yvalues[i] * exp(current_guess * xvalues[i])
        numerator_a = numerator_a + exp(current_guess*xvalues[i])*yvalues[i]  
        denominator_a = denominator_a + exp(2*current_guess*xvalues[i])
        mult = mult + exp(2*current_guess*xvalues[i]) * xvalues[i]
    ret = ret - (numerator_a/denominator_a) * mult
    return ret
    
def bisection(low, high, xvalues, yvalues, error_limit = 0.001, max_iteration = 50):
    step = 1
    mid = (low + high) / 2
    error = None

    while True:
        old_mid = mid
        mid = (low + high) / 2
        val_l = EquationForBisectionSolve(xvalues, yvalues, low)
        val_m = EquationForBisectionSolve(xvalues, yvalues, mid)
        if val_l * val_m < 0:
            high = mid
        else:
            low = mid
        if step > 1:
            error = abs(((mid - old_mid) / mid) * 100)
        if step > max_iteration or (step > 1 and error <= error_limit):
            return mid
        step += 1

# For Transformation
def linear_regression(xvalues, yvalues):
    # y = a_0 + a_1 * x
    # a_1 = (n*productsum - xsum * ysum) / (n*xsquaresum - (xsum)^2)
    # a_0 = yavg - a_1 * xavg
    
    n = xvalues.size
    xsum, xsquaresum, ysum, productsum = 0, 0, 0, 0
    for x in xvalues:
        xsum += x
        xsquaresum += x*x
    for y in yvalues:
        ysum += y
    for i in range(n):
        productsum += xvalues[i] * yvalues[i]
    
    a1 = (n*productsum - xsum * ysum) / (n*xsquaresum - (xsum)**2)
    a0 = ysum/n - a1 * (xsum / n)
    
    return (a0, a1)
        
def exponential_regression(xvalues, yvalues, transformation = False):
    # y = a*exp(bx)
    # a = sum_y_ebx / sum_e2bx
    # sum_x_y_ebx - a * sum_x_e2bx = 0, first solve for b, then you will get a (1)
    # terms are numbered from equation (1)
    
    a, b = None, None
    
    n = xvalues.size
    if not transformation:
        b = bisection(-50, 50, xvalues, yvalues)
        sum_y_ebx, sum_e2bx = 0, 0
        for i in range(n):
            sum_y_ebx = sum_y_ebx + yvalues[i] * exp(b*xvalues[i])
            sum_e2bx = sum_e2bx + exp(2*b*xvalues[i])
        a = sum_y_ebx / sum_e2bx
    else:
        # lny = ln(a) + bx
        # lny = z, lna = a0 in z = a_0 + a_1*x
        ret = linear_regression(xvalues, np.log(yvalues))
        a = exp(ret[0])
        b = ret[1]
        
    # Let's do the plot
    pt.plot(xvalues, yvalues, "ro")
    xvals_np = np.arange(0, 100, 0.01)
    pt.plot(xvals_np, a*(np.exp(b*xvals_np)), "c-")
    pt.legend(["Data Points", "Fitted Curve"])
    pt.title("Exponential Regression")
    pt.xlabel("x")
    pt.ylabel("y")
    pt.show()
    
    return (a, b)


print(exponential_regression(np.array([0, 1, 3, 5, 7, 9]), np.array([1.000, 0.891, 0.708, 0.562, 0.447, 0.355]), False))
        
        