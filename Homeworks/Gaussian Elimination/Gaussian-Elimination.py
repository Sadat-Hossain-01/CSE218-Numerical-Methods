# x^2 + y^2 + ax + by + c = 0 passes through (-2, 0), (-1, 7), (5, -1)
# Answer should be (a, b, c) = (-4, -6, -12)

import numpy as np

def GaussianElimination(A, B, pivot=True, showall=True):
    shape1, shape2 = np.shape(matA), np.shape(matB)
    assert shape1[0] == shape1[1] and shape1[0] == shape2[0] and shape2[1] == 1
    n = shape1[0]
    
    # Forward Elimination
    for step in range(1, n):
        if showall:
            print(f'Step {step}:')
        for sub_step in range(0, n-step):
            multiplier = A[sub_step+step][step-1]/A[step-1][step-1]
            A[sub_step+step] -= A[step-1] * multiplier
            B[sub_step+step] -= B[step-1] * multiplier
            if showall:
                print(f'Sub-step {sub_step+1}:\nA:\n{A}\nB:\n{B}\n')
        
    
    matAns = np.empty((n, 1))
    
    # Back Substitution
    matAns[n-1][0] = B[n-1][0]/A[n-1][n-1]
    
    
    return matAns
    
    
# n = int(input())
# matA = np.empty((n, n))
# for i in range(n):
#     matA[i] = [float(x) for x in input().split()]


# matB = np.reshape(np.array([float(x) for x in input().split()]), (n, 1))
matA = np.array([[-2, 0, 1], [-1, 7, 1], [5, -1, 1]], dtype='float64')
matB = np.array([[-4], [-50], [-26]], dtype='float64')
# print(matA, matB, sep='\n\n',end='\n\n')
print(GaussianElimination(matA, matB, showall=True))
    

    