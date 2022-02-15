import numpy as np
import matplotlib.pyplot as plt

# y=a0+a1x+a2x^2+a3x^3

def forwardElimination(A,B):
    row = np.size(A, 0)
    for i in range(row):
        max_value = abs(A[i][i])
        max_row=i #where the max value is found
        for j in range(i+1,row):
            if abs(A[j][i])>max_value:
                max_value=A[j][i]
                max_row=j
        for k in range(row):
            temp=A[max_row][k]
            A[max_row][k]=A[i][k]
            A[i][k]=temp  #for row exchange
        temp2=B[max_row]
        B[max_row]=B[i]
        B[i]=temp2
        for j in range(i+1,row):
            multiplier = float(A[j][i]) / A[i][i]
            for k in range(row):
                A[j][k]=A[j][k]-A[i][k]*multiplier
            B[j]=B[j]-B[i]*multiplier
    return A,B
def GaussianElimination(A,B):
    row=np.size(A,0)
    A,B=forwardElimination(A,B)
    sol=np.zeros(row,float)    #[0 0 0]
    for i in reversed(range(row)):
        #print(sol)
        dot_product = sol.dot(A[i])
        sol[i] = (B[i] - dot_product) / A[i][i]
        #sol[i] = f'{sol[i]:.4f}'
    return sol

def polinomialRegression(x,y):
    print(x, y)
    n=len(x)
    xi = 0
    xi2 = 0
    xi3 = 0
    xi4 = 0
    xi5=0
    xi6=0
    yi = 0
    yixi = 0
    xi2yi =0
    xi3yi=0
    for i in range(0,n):
        xi=xi+x[i]
        xi2=xi2+x[i]*x[i]
        xi3=xi3+x[i]**3
        xi4=xi4+x[i]**4
        xi5=xi5+x[i]**5
        xi6 = xi6 + x[i]**6
        yi=yi+y[i]
        yixi=yixi+y[i]*x[i]
        xi2yi=xi2yi+y[i]*(x[i]**2)
        xi3yi=xi3yi+y[i]*(x[i]**3)
    A=[n,xi,xi2,xi3,xi,xi2,xi3,xi4,xi2,xi3,xi4,xi5,xi3,xi4,xi5,xi6]
    B=[yi,yixi,xi2yi,xi3yi]
    A = np.array(A,dtype='float64')
    A = A.reshape(4, 4)
    B = np.array(B,dtype='float64')
    B.reshape(4, 1)
    print(A)
    print(B)
    sol = GaussianElimination(A, B)
    #sol = np.linalg.inv(A).dot(B)
    return sol

X = [0,10,20,30,40,50,60,70,80,90,100]
Y = [10.3,13.5,13.9,14.2,11.6,10.3,9.7,9.6,14.1,19.8,31.1]
x=np.array(X,dtype='float64')
y=np.array(Y,dtype='float64')
sol=polinomialRegression(x,y)
a0=sol[0]
a1=sol[1]
a2=sol[2]
a3=sol[3]
print(a0)
print(a1)
print(a2)
print(a3)
plt.scatter(x,y)
plt.plot(x,(a0+a1*x+a2*x*x+a3*x*x*x),"red")
plt.show()
print(a0+a1*110+a2*110**2+a3*110**3)