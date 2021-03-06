import numpy as np
import sympy as sp
np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})
# Book Problem E-3
# v = b_0 + b_1(t-t_0) + b_2(t-t_0)(t-t_1) + b_3(t-t_0)(t-t_1)(t-t_2)

xval = np.array([10, 15, 20, 22.5])
yval = np.array([227.04, 362.78, 517.35, 602.97])

def divided_difference(arr):
    sz = len(arr)
    ret = None
    if sz == 1:
        ret = yval[arr[0]]
    else:
        numerator = divided_difference(arr[0:sz-1]) - divided_difference(arr[1:sz])
        denominator = xval[arr[0]] - xval[arr[sz-1]]
        ret = numerator / denominator
    return ret
    
b = np.empty(4)
lst = list()
for i in range(4):
    lst.insert(0, i)
    print(lst)
    b[i] = divided_difference(lst)
        
print(b)

# so velocity at t=16 seconds
ans = 0
for term in range(0, 4):
    this = b[term]
    for j in range(term):
        print(f'Multiplying by 16-{xval[j]}')
        this *= (16 - xval[j])
    print(term, this)
    ans += this
print(f'at t = 16 seconds, velocity = {ans}')

# Distance covered from 11 to 16 seconds
t = sp.Symbol('t')
v = b[0] + b[1] * (t-xval[0]) + b[2] * (t-xval[0]) * (t-xval[1]) + b[3] * (t-xval[0]) * (t-xval[1]) * (t-xval[2])
s = sp.integrate(v)
print(f'v = {v}')
print(f's = {s}')
s = sp.lambdify(t, s)
print(f'Distance covered from 11 to 16 seconds = {s(16) - s(11)} m')

# Accelerationa t t=16
a = sp.diff(v)
print(f'a = {a}')
a = sp.lambdify(t, a)
print(f'Acceleration at t=16 second = {a(16)}')