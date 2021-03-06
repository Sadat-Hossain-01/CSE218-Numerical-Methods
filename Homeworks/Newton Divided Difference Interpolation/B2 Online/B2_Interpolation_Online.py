import numpy as np
import matplotlib.pyplot as pt
np.set_printoptions(suppress=True, formatter={'float': '{:0.4f}'.format})

def func(x, b, taken_x):
    return b[0] + b[1] * (x - taken_x[0]) + b[2] * (x-taken_x[0]) * (x-taken_x[1]) + b[3] * (x-taken_x[0]) * (x-taken_x[1]) * (x-taken_x[2])

def findNearestPoints(xvalues, yvalues, required, x):
    INF = 1000000000
    n = len(xvalues)
            
    right_idx = -1
    for i in range(n):
        if (xvalues[i] < x):
            continue
        if right_idx == -1:
            right_idx = i
            break
    left_idx = right_idx - 1
    taken_x = np.array([left_idx, right_idx])
    done = 2
    left_idx = left_idx - 1
    right_idx = right_idx + 1
    
    while done < required:
        ld, rd = INF, INF
        if left_idx >= 0:
            ld = abs(xvalues[left_idx] - x)
        if right_idx < n:
            rd = abs(xvalues[right_idx] - x)
        if ld < rd:
            taken_x = np.append(taken_x, left_idx)
            left_idx = left_idx - 1
        else:
            taken_x = np.append(taken_x, right_idx)
            right_idx = right_idx + 1
        done = done + 1

    taken_x = np.sort(taken_x)
    taken_y = np.empty(required)
    
    for i in range(required):
        taken_y[i] = yvalues[taken_x[i]]
        taken_x[i] = xvalues[taken_x[i]]
    return taken_x, taken_y

def newton_interpolation(xvalues, yvalues, order, x):
    EPS = 1e-8
    for i in range (len(xvalues)):
        if abs(xvalues[i] - x) <= EPS:
            return yvalues[i]
    required = order + 1
    taken_x, taken_y = findNearestPoints(xvalues, yvalues, required, x)
    difference_table = np.empty((required, required))

    for length in range(0, required):
        for row in range(0, required):
            if length == 0:
                difference_table[row][row] = taken_y[row]
            elif row + length < required:
                difference_table[row+length][row] = (difference_table[row+length][row+1] - difference_table[row+length-1][row]) / (taken_x[row+length] - taken_x[row])

    # print(difference_table)
    b = np.empty(required)
    for i in range(required):
        b[i] = difference_table[i][0]

    ans = 0
    for term in range(0, required):
        this = b[term]
        for j in range(term):
            this *= (x - taken_x[j])
        ans += this
    
    # xx = np.arange(taken_x[0], taken_x[required-1], 0.01)
    # pt.plot(xx, func(xx, b, taken_x), "g")
    xlist = list();
    pt.grid(True, which="both")
    for i in range(required):
        xlist.append(taken_x[i])
        pt.plot(taken_x[i], taken_y[i], "co")
    pt.plot(x, ans, "co")
    pt.plot(np.array(xlist), func(np.array(xlist), b, taken_x))
    pt.legend(["Estimation"])
    pt.show()   
    
    return ans

xvalues = list()
yvalues = list()
with open("gene.txt", "r") as file:
    for line in file:
        xvalues.append(float(line.split()[0]))
        yvalues.append(float(line.split()[1]))
print(newton_interpolation(xvalues, yvalues, 3, 17))


            
            




