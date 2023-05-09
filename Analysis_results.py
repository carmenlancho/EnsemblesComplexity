#################  SCRIPT TO ANALYZE RESULTS FROM BAGGING ####################


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





def plot_acc_ensemble(data,name):
    plt.plot(data.loc[data.weights == 'CLD','n_ensemble'],
             data.loc[data.weights == 'CLD','accuracy_mean'], c = 'blue', label = 'CLD')
    plt.plot(data.loc[data.weights == 'DCP','n_ensemble'],
             data.loc[data.weights == 'DCP','accuracy_mean'], c = 'green', label = 'DCP')
    plt.plot(data.loc[data.weights == 'LSC','n_ensemble'],
             data.loc[data.weights == 'LSC','accuracy_mean'], c = 'lime', label = 'LSC')
    plt.plot(data.loc[data.weights == 'TD_U','n_ensemble'],
             data.loc[data.weights == 'TD_U','accuracy_mean'], c = 'orange', label = 'TD_U')
    plt.plot(data.loc[data.weights == 'N2','n_ensemble'],
             data.loc[data.weights == 'N2','accuracy_mean'], c = 'purple', label = 'N2')
    plt.plot(data.loc[data.weights == 'F1','n_ensemble'],
             data.loc[data.weights == 'F1','accuracy_mean'], c = 'cyan', label = 'F1')
    plt.plot(data.loc[data.weights == 'Uniform','n_ensemble'],
             data.loc[data.weights == 'Uniform','accuracy_mean'], c = 'k', label = 'Uniform')
    plt.plot(data.loc[data.weights == 'N1','n_ensemble'],
             data.loc[data.weights == 'N1','accuracy_mean'], c = 'pink', label = 'N1')
    plt.plot(data.loc[data.weights == 'kDN','n_ensemble'],
             data.loc[data.weights == 'kDN','accuracy_mean'], c = 'gold', label = 'kDN')
    plt.plot(data.loc[data.weights == 'Hostility','n_ensemble'],
             data.loc[data.weights == 'Hostility','accuracy_mean'], c = 'crimson', label = 'Hostility')
    plt.legend(loc='lower center', bbox_to_anchor=(0.55, 0.0),
          ncol=4, fancybox=False, shadow=False)
    plt.title(name)
    plt.show()

    return

#######################################################################
#################    More weight in hard instances WITH RANKING   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' not in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)



#######################################################################
#################    More weight in easy instances WITH RANKING    #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' not in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)

path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,159])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble']).T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)





#######################################################################
#################    More weight in hard instances WITH 1/n +   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' in filename and 'classes' not in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)



#######################################################################
#################    More weight in easy instances WITH 1/n +     #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' in filename  and 'classes' not in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)




#######################################################################
#################    More weight in hard instances WITH RANKING classes   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)



#######################################################################
#################    More weight in easy instances WITH RANKING  classes   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)





#######################################################################
#################    More weight in hard instances WITH 1/n + classes  #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' in filename and 'classes' in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)



#######################################################################
#################    More weight in easy instances WITH 1/n + classes    #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' in filename  and 'classes' in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)




#######################################################################
#################    More weight in frontier    #################
#######################################################################
# We exclude LSC because it generally offers complexity values higher than 0.9
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'frontier' in filename and 'Aggregated' in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)









