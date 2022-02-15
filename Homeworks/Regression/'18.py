import numpy as np
import matplotlib.pyplot as pt

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

# A1

xvalues = np.array([4.96, 3.02, 12.36, 10.6, 4.45, 6.4, 1.58, 2.9, 7.55, 0.41])
yvalues = np.array([3.98, 2, 5.28, 4.09, 2.75, 3.04, 3.67, 3.45, 4.35, 1.53])
ret = linear_regression(np.exp(xvalues)/xvalues, yvalues/xvalues)
a = ret[0]
b = ret[1]
print(a, b)

# Let's do the plot
n = xvalues.size
pt.plot(xvalues, yvalues, "ro")
xvals_np = np.arange(0.4, 12.4, 0.01)
pt.plot(xvals_np, a * xvals_np + b * np.exp(xvals_np), "c-")
pt.legend(["Data Points", "Regressed Curve"])
pt.title("Regression")
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()

# A2

ret = linear_regression(1/xvalues**2, 1/yvalues)
a = 1 / ret[0]
b = ret[1] * a
print(a, b)

# Let's do the plot
n = xvalues.size
pt.plot(xvalues, yvalues, "ro")
xvals_np = np.arange(0.4, 12.4, 0.01)
pt.plot(xvals_np, (a * xvals_np * xvals_np) / (b + xvals_np * xvals_np), "c-")
pt.legend(["Data Points", "Regressed Curve"])
pt.title("Regression")
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()