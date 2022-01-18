import numpy as np
import matplotlib.pyplot as pt

np.set_printoptions(suppress=True, formatter={'float': '{:0.4f}'.format})

def findNearestPoints(xvalues, yvalues, requiredPoints, x):
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
    
    while done < requiredPoints:
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
    taken_y = np.empty(requiredPoints)
    
    for i in range(requiredPoints):
        taken_y[i] = yvalues[taken_x[i]]
        taken_x[i] = xvalues[taken_x[i]]
    return taken_x, taken_y



def lagrange_interpolation(taken_x, taken_y, order, x):
    required = order + 1
    ans = 0
    b = np.empty(required)
    for i in range(required):
        b[i] = 1
        for j in range(0, required):
            if i == j:
                continue
            b[i] *= (x - taken_x[j]) / (taken_x[i] - taken_x[j])
        ans += b[i] * taken_y[i]
    return ans


days = list()
index = list()


with open("stock.txt", "r") as file:
    idx = 0
    for line in file:
        if idx > 0:
            # print(line)
            days.append(float(line.split()[0]))
            index.append(float(line.split()[1]))
        idx = idx + 1
# print(days, index)

x = float(input())

takenXCubic, takenYCubic = findNearestPoints(days, index, 4, x)
cubicAns = lagrange_interpolation(takenXCubic, takenYCubic, 3, x)
takenXQuadratic, takenYQuadratic = findNearestPoints(days, index, 3, x)
quadraticAns = lagrange_interpolation(takenXQuadratic, takenYQuadratic, 2, x)
print(f'Answer using cubic interpolation: {cubicAns}')
print(f'Answer using quadratic interpolation: {quadraticAns}')
error = abs((cubicAns - quadraticAns) / cubicAns) * 100
print(f'Absolute Relative Approximate Error: {error}%')

pt.plot(days, index, "g", [x,x], [-1.5, cubicAns], "b")
pt.legend(["Curve", "Estimation"])
pt.grid(True, which="both")
pt.show()


