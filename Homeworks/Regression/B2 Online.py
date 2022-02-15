import matplotlib.pyplot as pt
import numpy as np

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


real_x = np.array([0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21])
real_y = np.array([1, 1.03, 1.06, 1.38, 2.09, 3.54, 6.41, 12.6, 22.1, 39.05, 65.32, 99.78])

passed_y = np.log(real_y)
ret = linear_regression(real_x, passed_y)
# y = a*exp(bx)
a = np.exp(ret[0])
b = ret[1]
print(a, b)
    
# Let's do the plot
pt.plot(real_x, real_y, "ro")
n = real_x.size
xvals_np = np.arange(0, 0.22, 0.01)
pt.plot(xvals_np, a*np.exp(b*xvals_np), "c-")
pt.legend(["Data Points", "Fitted Curve"])
pt.title("Exponential Model Regression")
pt.xlabel("x")
pt.ylabel("y")
pt.show()

print("The probability of crashing " + str(a*np.exp(b*0.16)))


