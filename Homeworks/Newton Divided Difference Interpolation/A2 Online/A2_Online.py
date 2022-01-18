import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

np.set_printoptions(suppress=True, formatter={'float': '{:0.4f}'.format})

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

def newton_interpolation(taken_x, taken_y, order, x):
    EPS = 1e-8
    required = order + 1
    for i in range (required):
        if abs(xvalues[i] - x) <= EPS:
            return yvalues[i]
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
    return ans

data = pd.read_csv("dissolveO2.csv").to_numpy()
print(data)

# xxx = np.array(data['month_number'])
# yy1 = np.array(data['month_number'])
# yy2 = np.array(data['month_number'])

xvalues = list()
y1values = list()
y2values = list()
for line in data:
    xvalues.append(line[0])
    y1values.append(line[1])
    y2values.append(line[2])
    
# inp = float(input())
inp = 26
taken_x, taken_y1 = findNearestPoints(xvalues, y1values, 5, inp)
taken_x, taken_y2 = findNearestPoints(xvalues, y2values, 5, inp)
print(taken_x)
one = newton_interpolation(taken_x, taken_y1, 4, inp)
two = newton_interpolation(taken_x, taken_y2, 4, inp)
print(one, two)

pt.grid(True, which="both")
pt.plot(xvalues, y1values, xvalues, y2values, [inp, inp], [0, one], [inp, inp], [0, two])
pt.legend(["1bar", "2bar", "estimate1bar", "estimate2bar"])
pt.show()

    
