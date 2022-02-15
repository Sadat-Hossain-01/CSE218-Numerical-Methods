import pandas as pd

days = list()
index = list()

with open("stock.txt", "r") as file:
    idx = 0
    
    for line in file:
        print(line)
        print(line.split())
        
        if idx > 0:
            days.append(float(line.split()[0]))
            index.append(float(line.split()[1]))
        
        idx = idx + 1
        
