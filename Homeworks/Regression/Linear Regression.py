import matplotlib.pyplot as pt
import numpy as np

def linear_regression(xvalues, yvalues):
    # y = a_0 + a_1 * x
    # a_1 = (n*productsum - xsum * ysum) / (n*xsquaresum - (xsum)^2)
    # a_0 = yavg - a_1 * xavg
    
    n = len(xvalues)
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
    
    # Let's do the plot
    pt.plot(xvalues, yvalues, "ro")
    xvals_np = np.arange(0, xvalues[n-1] + 1, 0.01)
    pt.plot(xvals_np, a0 + a1*xvals_np, "c-")
    pt.legend(["Data Points", "Fitted Curve"])
    pt.title("Linear Regression")
    pt.xlabel("x")
    pt.ylabel("y")
    pt.show()
    
    return (a0, a1)

print(linear_regression([0.698132, 0.959931, 1.134464, 1.570796, 1.919862], [0.188224, 0.209138, 0.230052, 0.250965, 0.313707]))