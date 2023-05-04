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
    # file = data_file
    # name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data[['x1', 'x2']].to_numpy()
    y = data[['y']].to_numpy()

