import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_path = os.getcwd()
path_csv = os.chdir(root_path+'/datasets')

file = 'XOR_6000_estandarizado.csv'
# 'Dataset1_6000_estandarizado.csv', 'Dataset2_6000_estandarizado., 'Dataset3_6000_estandarizado.csv',
# 'Dataset4_6000_estandarizado.csv'
# 'Dataset5_6000_estandarizado.csv', 'Dataset6_6000_estandarizado.csv', 'Dataset7_6000_estandarizado.csv'
# 'Dataset9_6000_estandarizado.csv', 'Dataset10_6000_estandarizado.csv', 'XOR_6000_estandarizado.csv'
# 'Moon_train_estandarizado.csv'
data = pd.read_csv(file)
# Para leerlo y pasarlo a numpy
y = data['y'].to_numpy()
X = data.iloc[:, 0:-1].to_numpy()
data.columns = ['x1', 'x2', 'classes']


# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.classes == 1)
idx_0 = np.where(data.classes == 0)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.show()



