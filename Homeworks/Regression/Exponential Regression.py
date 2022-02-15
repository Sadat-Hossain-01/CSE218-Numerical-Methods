import matplotlib.pyplot as pt
import numpy as np

# For Transformation
def linear_regression(xvalues, yvalues):
    # y = a + b * x
    # b = (n*productsum - xsum * ysum) / (n*xsquaresum - (xsum)^2)
    # a = yavg - b * xavg
    
    n = xvalues.size
    xsum, xsquaresum, ysum, xyproductsum = 0, 0, 0, 0
    
    for x in xvalues:
        xsum += x
        xsquaresum += x * x
        
    for y in yvalues:
        ysum += y
        
    for i in range(n):
        xyproductsum += xvalues[i] * yvalues[i]
    
    b = (n * xyproductsum - xsum * ysum) / (n * xsquaresum - xsum * xsum)
    a = ysum / n - b * (xsum / n)
    
    return (a, b)

# Bisection Required for Exponential Model

def EquationForBisectionSolve(xvalues, yvalues, current_guess):
    # y = a * exp(bx)
    # a = sum_y*exp(bx) / sum_exp(2bx)
    # sum_xy*exp(bx) - a * sum_x*exp(2bx) = 0, substitute a's value here, solve for b by bisection method, then you will get a
    # numerator_a = sum_y*exp(bx)
    # denominator_a = sum_exp(2bx)
    # mult = sum_x*exp(2bx)
    
    ret = 0
    numerator_a, denominator_a, mult = 0, 0, 0
    n = xvalues.size
    for i in range(n):
        ret += xvalues[i] * yvalues[i] * np.exp(current_guess * xvalues[i])
        numerator_a += np.exp(current_guess * xvalues[i]) * yvalues[i]  
        denominator_a += np.exp(2 * current_guess * xvalues[i])
        mult += np.exp(2 * current_guess * xvalues[i]) * xvalues[i]
    ret -= (numerator_a / denominator_a) * mult
    
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
        
def exponential_regression(xvalues, yvalues):
    # y = a * exp(bx)
    # a = sum_y*exp(bx) / sum_exp(2bx)
    # sum_xy*exp(bx) - a * sum_x*exp(2bx) = 0, substitute a's value here, solve for b by bisection method, then you will get a
    
    a, b = None, None
    
    n = xvalues.size
    b = bisection(-50, 50, xvalues, yvalues)
    sum_y_ebx, sum_e2bx = 0, 0
    
    for i in range(n):
        sum_y_ebx += yvalues[i] * np.exp(b*xvalues[i])
        sum_e2bx += np.exp(2*b*xvalues[i])
    
    a = sum_y_ebx / sum_e2bx
    
    return (a, b)


xvalues = np.array([0, 1, 3, 5, 7, 9])
yvalues = np.array([1.000, 0.891, 0.708, 0.562, 0.447, 0.355])

# Without Transformation
ret = exponential_regression(xvalues, yvalues) 
a, b = ret[0], ret[1]
print(a, b)

# Let's do the plot
pt.plot(xvalues, yvalues, "ro")
xvals_np = np.arange(0, 100, 0.01)
pt.plot(xvals_np, a * (np.exp(b * xvals_np)), "c-")
pt.legend(["Data Points", "Fitted Curve"])
pt.title("Exponential Regression without Transformation")
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()      

# With Transformation
ret = linear_regression(xvalues, np.log(yvalues))
a, b = np.exp(ret[0]), ret[1]
print(a, b)

# Let's do the plot
pt.plot(xvalues, yvalues, "go")
xvals_np = np.arange(0, 100, 0.01)
pt.plot(xvals_np, a * (np.exp(b * xvals_np)), "y-")
pt.legend(["Data Points", "Fitted Curve"])
pt.title("Exponential Regression with Transformation")
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()                         
        