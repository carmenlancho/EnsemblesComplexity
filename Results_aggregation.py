#####################################################################################################
####### In this script we aggregate the results per fold to have global values of performance #######
#####################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


root_path = os.getcwd()



path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Difficult' in filename):
        total_name_list.append(filename)


# total_name_list = ['Bagging_Data13_MoreWeightDifficultInstances.csv']



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    data = pd.read_csv(file)


## Agregado
cols_numeric = data.select_dtypes([np.number]).columns[2:]

aa = data.groupby(['n_ensemble','weights'], as_index=False)[cols_numeric].mean()
bb = data.groupby(['n_ensemble','weights'], as_index=False)[cols_numeric].std()

n_ensemble_list = np.unique(data['n_ensemble']).tolist()
weights_list = np.unique(data['weights']).tolist()

for n_i in n_ensemble_list:
    print(n_i)
    for w in weights_list:
        print(w)
        condition = (data.n_ensemble == n_i) & (data.weights == w)
        data_pack = data.loc[condition]
        data_pack['confusion_matrix'].to_numpy()
        data_pack['confusion_matrix']


        import re

        dd = data_pack['confusion_matrix'][399]
        dd2 = [int(s) for s in re.findall(r'\b\d+\b', dd)]
        dd2 = np.array(dd2)
        dd2.shape = (2, 2)
        dd2 + dd2

        hh = data_pack['Boots_N2_class'][399]
        np.fromstring(data_pack['Boots_N2_class'][399], dtype=int, sep=',')
        np.fromstring(hh.strip('[]'), sep=',')


