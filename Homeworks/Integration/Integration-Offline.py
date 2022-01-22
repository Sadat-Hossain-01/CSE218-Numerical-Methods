import numpy as np
import matplotlib.pyplot as pt

C_initial = 1.22e-4

def func(x, C_me = 5e-4):
    numerator = -1 * (6.73*x + 6.725e-8 + 7.26e-4 * C_me)
    denominator = 3.62e-12 * x + 3.908e-8 * x * C_me
    return numerator/denominator

def integration_trapezoid(func, a, b, n):
    h = (b - a) / n
    ans = 0
    ans = ans + func(a)
    for i in range(1, n):
        ans = ans + 2 * func(a + i * h)
    ans = ans + func(b)
    ans = ans * (h / 2)
    return ans

def integration_simpsons_single(func, a, b):
    h = (b - a) / 2
    ans = func(a) + 4 * func((a + b) / 2) + func(b)
    ans = ans * (h / 3)
    return ans

def integration_simpsons_multiple(func, a, b, n):
    assert n % 2 == 0
    ans = 0
    h = (b - a) / n
    for i in range(0, n, 2):
        ans = ans + integration_simpsons_single(func, a + i * h, a + (i + 2) * h)
    return ans
    
# analytical_answer = 17738697.18

# Problem 1
previous_answer, error = None, None
print('Trapezoidal Method:')
for n in range (1, 6):
    this_ans = integration_trapezoid(func, C_initial, 0.5 * C_initial, n)
    print(f'n = {n} Result: {format(this_ans, ".6f")} seconds Error: ', end='')
    if n > 1:
        error = abs((this_ans - previous_answer) / this_ans) * 100
        print(f'{format(error, ".6f")}%')
    else:
        print('N/A')
    previous_answer = this_ans

# Problem 2
print('Simpson\'s 1/3rd Rule:')
for n in range (1, 6):
    this_ans = integration_simpsons_multiple(func, C_initial, 0.5 * C_initial, 2 * n)
    print(f'n = {n} Result: {format(this_ans, ".6f")} seconds Error: ', end='')
    if n > 1:
        error = abs((this_ans - previous_answer) / this_ans) * 100
        print(f'{format(error, ".6f")}%')
    else:
        print('N/A')
    previous_answer = this_ans
    
# Problem 3
x = np.array([1.22, 1.20, 1.0, 0.8, 0.6, 0.4, 0.2]) * 1e-4
y = integration_simpsons_multiple(func, C_initial, x, 10) / 3600
# print(y)
pt.plot(x, y, "p-")
pt.xlabel("Concentration (moles/cc)")
pt.ylabel("Time (Hour)")
pt.show()
    
    
    

    