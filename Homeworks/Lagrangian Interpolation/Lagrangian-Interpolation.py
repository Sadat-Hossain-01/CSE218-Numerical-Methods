import numpy as np

b = np.ones(3, dtype='float64')
xval = np.array([10, 15, 20])
yval = np.array([227.04, 362.78, 517.35])

ans = 0 # v at t=16
for i in range(3):
    b[i] = 1
    for j in range(0, 3):
        if i == j:
            continue
        b[i] *= (16 - xval[j]) / (xval[i] - xval[j])
    ans += b[i] * yval[i]
        
print(b)
print(ans)
