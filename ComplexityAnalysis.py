###### Complexity analysis of the artificial datasets

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_path = os.getcwd()
path_csv = os.chdir(root_path+'/Results_Complexity_InstanceLevel')


# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


cols_name = ['dataset','Hostility', 'kDN', 'DS', 'DCP', 'TD_U', 'TD_P', 'MV', 'CB', 'CLD',
       'N1', 'N2', 'LSC', 'LSradius', 'H', 'U', 'F1', 'F2', 'F3', 'F4']
data_total = pd.DataFrame(columns=cols_name)

for data_file in total_name_list:
    print(data_file)
    file = data_file

    file_name = data_file[33:-4]
    data = pd.read_csv(file,index_col=0)
    data['dataset'] = file_name
    data_total = pd.concat([data_total, data], ignore_index=True)


data_total


sns.boxplot(data=data_total, x="CLD", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="DCP", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="F1", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="LSC", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="N1", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="N2", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="TD_U", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="kDN", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="Hostility", y="dataset")
plt.show()