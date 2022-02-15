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

def concised_linear_regression(xvalues, yvalues):
    # y = a + b * x
    # b = (n*productsum - xsum * ysum) / (n*xsquaresum - (xsum)^2)
    # a = yavg - b * xavg
    
    n = xvalues.size
    xsum, xsquaresum, ysum, xyproductsum = np.sum(xvalues), np.sum(xvalues*xvalues), np.sum(yvalues), np.sum(xvalues*yvalues)
    
    b = (n * xyproductsum - xsum * ysum) / (n * xsquaresum - xsum * xsum)
    a = ysum / n - b * (xsum / n)
    
    return (a, b)


xvalues = np.array([0.698132, 0.959931, 1.134464, 1.570796, 1.919862])
yvalues = np.array([0.188224, 0.209138, 0.230052, 0.250965, 0.313707])
ret = linear_regression(xvalues, yvalues)

# Let's do the plot
n = xvalues.size
pt.plot(xvalues, yvalues, "ro")
xvals_np = np.arange(xvalues[0] - 0.2, xvalues[n-1] + 0.2, 0.01)
pt.plot(xvals_np, ret[0] + ret[1] * xvals_np, "c-")
pt.legend(["Data Points", "Regressed Curve"])
pt.title("Linear Regression")
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()