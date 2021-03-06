import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:0.6f}'.format})
# Book Problem P-6
# v = b_0 + b_1 (t-t_0) + b_2 (t-t_0)(t-t_1)

xval = np.array([10, 15, 20])
yval = np.array([227.04, 362.78, 517.35])

def divided_difference(arr):
    sz = len(arr)
    if sz == 1:
        return yval[arr[0]]
    else:
        numerator = divided_difference(arr[0:sz-1]) - divided_difference(arr[1:sz])
        denominator = xval[arr[0]] - xval[arr[sz-1]]
        return numerator / denominator
    
b = np.empty(3)
lst = list()
for i in range(3):
    lst.insert(0, i)
    print(lst)
    b[i] = divided_difference(lst)
        
print(b)

# so velocity at t=16 seconds
ans = 0
for term in range(0, 3):
    this = b[term]
    for j in range(term):
        print(f'Multiplying by 16-{xval[j]}')
        this *= (16 - xval[j])
    print(term, this)
    ans += this
print(f'at t = 16 seconds, velocity = {ans}')

