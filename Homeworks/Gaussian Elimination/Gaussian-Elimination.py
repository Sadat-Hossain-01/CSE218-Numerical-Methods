# x^2 + y^2 + ax + by + c = 0 passes through (-2, 0), (-1, 7), (5, -1)
# Answer for the above sample should be (a, b, c) = (-4, -6, -12)

# -2a + c = -4
# -a + 7b + c = -50
# 5a - b + c = -26


import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:0.4f}'.format})


def GaussianElimination(A, B, pivot=True, showall=True):
    shape1, shape2 = np.shape(matA), np.shape(matB)
    assert shape1[0] == shape1[1] and shape1[0] == shape2[0] and shape2[1] == 1
    n = shape1[0]

    # Forward Elimination
    for step in range(1, n):
        if pivot:
            maxElement = abs(A[step-1][step-1])
            idx = step - 1
            for i in range(step, n):
                if abs(A[i][step-1]) > maxElement:
                    maxElement = abs(A[i][step-1])
                    idx = i
            A[[step-1, idx]] = A[[idx, step-1]]
            B[[step-1, idx]] = B[[idx, step-1]]
            # A[[l,r], a:b] actually means the submatrix containing A[l][a] to A[l][b-1] and A[r][a] to A[r][b-1]
        if showall:
            print(f'Step {step}:')
        for sub_step in range(0, n-step):
            assert not A[step-1][step-1] == 0
            multiplier = A[sub_step+step][step-1]/A[step-1][step-1]
            A[sub_step+step] -= A[step-1] * multiplier
            B[sub_step+step] -= B[step-1] * multiplier
            if showall:
                print(f'Sub-step {sub_step+1}:\nA:\n{A}\nB:\n{B}\n')

    matAns = np.empty((n, 1))

    # Back Substitution
    assert not A[n-1][n-1] == 0
    matAns[n-1][0] = B[n-1][0]/A[n-1][n-1]
    for i in range(n-2, -1, -1):
        assert i >= 0
        sum = 0
        for j in range(i+1, n):
            sum += A[i][j] * matAns[j]
        assert not A[i][i] == 0
        matAns[i] = (B[i][0]-sum)/A[i][i]

    return matAns


# n = int(input())
# matA = np.empty((n, n))
# matB = np.empty((n, 1))
# for i in range(n):
#     matA[i] = [float(x) for x in input().split()]

# for i in range(n):
#     matB[i][0] = float(input())

n = 3
matA = np.array([[25, 5, 1], [64, 8, 1], [144, 12, 1]], dtype='float64')
matB = np.array([[106.8], [177.2], [279.2]], dtype='float64')
# print(matA, matB, sep='\n\n', end='\n\n')

ret = GaussianElimination(matA, matB)
print('Solution Vector:')
for i in range(n):
    print(f'{ret[i][0]:.4f}')
