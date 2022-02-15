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


xvalues = np.array([0,0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21], dtype='float64')
yvalues = np.array([1,1.03,1.06,1.38,2.09,3.54,6.41,12.6,22.1,39.05,65.32,99.78], dtype='float64')

ret = concised_linear_regression(xvalues, np.log(yvalues))
a = np.exp(ret[0])
b = ret[1]
print(a, b)

# Let's do the plot
n = xvalues.size
pt.plot(xvalues, yvalues, "ro")
xvals_np = np.arange(0, 0.25, 0.01)
pt.plot(xvals_np, a * np.exp(b * xvals_np), "c-")
pt.legend(["Data Points", "Regressed Curve"])
pt.title("Linear Regression")
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()