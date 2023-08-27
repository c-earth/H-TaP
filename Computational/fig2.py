import numpy as np
import matplotlib.pyplot as plt

pris_file = 'D:/python_project/TaP/pristine.csv'
dope_file = 'D:/python_project/TaP/withH.csv'

pris_data = np.genfromtxt(pris_file, delimiter = ',', skip_header = 1)[:, -1]
dope_data = np.genfromtxt(dope_file, delimiter = ',', skip_header = 1)[:, -1]

plt.figure('pris', figsize = (8, 7), dpi = 100)
plt.plot(pris_data, '.')
# plt.show()

# plt.figure('dope', figsize = (8, 7), dpi = 100)
plt.plot(dope_data, '.')
plt.show()