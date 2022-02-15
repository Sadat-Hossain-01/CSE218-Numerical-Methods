import pandas as pd
import numpy as np

data = pd.read_csv('dissolveO2.csv')

# The following simply prints the data
print(data)

# The following converts the data to a 2D matrix/table
print(data.to_numpy())

# The following creates np_arrays that can be used for further computations
temperature = np.array(data['temperature'])
s1 = np.array(data['solubility_1'])
s2 = np.array(data['solubility_2'])

print(temperature)
print(s1)
print(s2)
