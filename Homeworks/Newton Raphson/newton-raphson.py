#The down the toilet problem from slides

from math import fabs

def evaluate(x):
    return x**3 - 0.165*x**2 + 3.993e-4

def derivative(x):
    return 3*(x**2) - 0.33*x 

def newton_raphson(guess, MAX_ITERATION, ERROR_LIMIT):
    step = 1
    x = guess
    old_x = guess
    error = None
    while True:
        old_x = x
        if x == 0:
            print("Division by zero error")
            return x, step, error
        
        x = old_x - evaluate(x)/derivative(x)
        
        
        if step > 1:
            error = fabs(((x - old_x) / x ) * 100)
            
        if step > MAX_ITERATION or (step > 1 and error < ERROR_LIMIT):
            return x, step, error
        
        step += 1
        
print(newton_raphson(0.02, 25, 0.05))