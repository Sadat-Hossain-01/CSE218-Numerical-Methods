import numpy as np
import matplotlib.pyplot as pt
np.set_printoptions(suppress=True, formatter={'float': '{:0.15f}'.format})

def GaussianElimination(A, B, pivot=True, showall=False):
    shape1, shape2 = np.shape(A), np.shape(B)
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

    matAns = np.empty((n, 1), dtype='float64')

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


def polynomial_regression(xvalues, yvalues, m):
    n = xvalues.size
    
    matA = np.empty((m + 1, m + 1))
    
    for row in range(m + 1):
        for col in range(m + 1):
            matA[row][col] = np.sum(xvalues**(row + col))
            
    matB = np.zeros((m + 1, 1), dtype='float64')
    

    for degree in range(0, m + 1):
        matB[degree][0] = np.sum((xvalues**degree) * yvalues)
        
    # print(matA, end='\n\n')
    # print(matB, end='\n\n')
         
    all_coeffs = GaussianElimination(matA, matB)
        
    return all_coeffs


# Let's do the plot
m = 2
xvalues = np.array([80, 40, -40, -120, -200, -280, -340], dtype = 'float64')
yvalues = np.array([6.47, 6.24, 5.72, 5.09, 4.30, 3.33, 2.45], dtype='float64') * 1e-6
n = xvalues.size
all_coeffs = polynomial_regression(xvalues, yvalues, m)
    
pt.plot(xvalues, yvalues, "ro")

xvals_np = np.arange(-400, 100, 0.1)
yplot = np.zeros((n))
for i in range(m + 1):
    yplot += all_coeffs[i][0] * np.power(xvalues, i)
pt.plot(xvalues, yplot, "c-")

pt.legend(["Data Points", "Fitted Curve"])
pt.title("Polynomial Regression of Order " + str(m))
pt.xlabel("x")
pt.ylabel("y")
pt.grid(True, which='both')
pt.show()            