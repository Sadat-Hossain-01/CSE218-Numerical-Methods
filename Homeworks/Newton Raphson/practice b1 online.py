import math
import matplotlib.pyplot as pt
import numpy as np

PI = math.acos(-1.0)

# f(h) = PI*(h^3) - 12*PI*(h^2) + 15

def derivative(x):
    return 3*PI*(x**2) - 24*PI*x

def func(x):
    return PI*(x**3) - 12*PI*(x**2) + 15

def plot():
    xvalues = np.arange(0, 1.5, 0.01)
    pt.plot(xvalues, func(xvalues))
    pt.axhline(y=0)
    pt.grid(True, which='both')
    pt.show()

def newton_raphson(guess):
    new_x = None
    old_x = guess
    step = 1
    error = None
    
    while True:
        try:
            new_x = old_x - func(old_x) / derivative(old_x)
        except:
            print("Division by zero error")
            return old_x
        
        if step > 1:
            error = math.fabs((new_x - old_x) / new_x) * 100
        
        if step == 1:
            print(f'Step {step}: N/A\n')
        else:
            print(f'Step {step}: {format(error, ".6f")}%\n')
        
        if step > 1 and error < 0.05:
            return old_x
        
        old_x = new_x
        step += 1
        
def bisection(lguess, hguess):
    low = lguess
    high = hguess
    step = 1
    old_mid = (low + high) / 2
    error = None
    
    while True:
        mid = (low + high) / 2
        
        if func(low) * func(mid) < 0:
            high = mid
        else:
            low = mid
            
        if step > 1:
            error = math.fabs(((old_mid - mid) / mid) * 100)
            
        if step == 1:
            print(f'Step {step}: N/A\n')
        else:
            print(f'Step {step}: {format(error, ".6f")}%\n')
            
        if step > 1 and error < 0.05:
            return mid
            
        step += 1
        old_mid = mid
        
    
plot()  
print(newton_raphson(0.6),end='\n\n')
print(bisection(0.4, 0.8))
    