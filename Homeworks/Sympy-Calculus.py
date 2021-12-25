from sympy import *
x = Symbol('x')
f = (x-4)*(x+8)
f_prime = f.diff(x)
print(f)
print(f_prime)
f_prime = lambdify(x, f_prime) #Converts from expression to fast calculable expression in which you can put variable values (like x = 5)
print(f_prime(2))

f_int = integrate(f)
print(f)
print(f_int)
f_int = lambdify(x, f_int)
print(f_int(1))