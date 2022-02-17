import numpy as np
import matplotlib.pyplot as pt

def linear_regression(xvalues, yvalues):
    # y = a + b * x
    # b = (n*productsum - xsum * ysum) / (n*xsquaresum - (xsum)^2)
    # a = yavg - b * xavg
    
    n = xvalues.size
    xsum, xsquaresum, ysum, xyproductsum = np.sum(xvalues), np.sum(xvalues*xvalues), np.sum(yvalues), np.sum(xvalues*yvalues)
    
    b = (n * xyproductsum - xsum * ysum) / (n * xsquaresum - xsum * xsum)
    a = ysum / n - b * (xsum / n)
    
    return (a, b)

xvalues = np.array([0, 5, 10, 15, 20, 25, 30], dtype='float64')
yvalues = np.array([1000, 550, 316, 180, 85, 56, 31], dtype='float64')

# Guessing the model

pt.plot(xvalues, yvalues, "ro")
pt.legend(["Data Points"])
pt.xlabel("Time (Hours)")
pt.ylabel("Amount (mg)")
pt.title("Plot for Guessing the Model")
pt.show()    

# So, the best model is exponential
# y = a * exp(bx)
# lny = lna + bx

ret = linear_regression(xvalues, np.log(yvalues))
a = np.exp(ret[0])
b = ret[1]

# print(f'a = {a}, b = {b}')
print(f'The best fitting model is the exponential model, y = a * e^(bx)\ny = {a:0.6} * e^({b:0.6}*x)')
print(f'where x and y represent hours passed and amount of drug (in mg) respectively.')

# Doing the Combined Plot
n = xvalues.size
pt.plot(xvalues, yvalues, "go")
xvals_np = np.arange(0, 41, 0.001)
pt.plot(xvals_np, a * np.exp(b * xvals_np), "c-")
pt.plot(40, a * np.exp(b * 40), "bo")
pt.legend(["Data Points", "Regressed Curve", "Predicted Value after 40hrs"])
pt.title("Final Plot")
pt.xlabel("Time (Hours)")
pt.ylabel("Amount (mg)")
pt.grid(True, which='both')
pt.show()

print(f'The predicted amount of drug in body after 40 hours is {(a * np.exp(b * 40)):0.6} mg.')


