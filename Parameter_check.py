#################################################################################################
####         Code to evaluate the complexity of the training samples for different values  ######
####                of the parameters to decide the upper limits of alpha and s           #######
#################################################################################################


################### SCRIPT TO ANALYZE COMPLEXITY OF THE BOOTSTRAP SAMPLES WITH EVERY STRATEGY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


root_path = os.getcwd()





path_csv = os.chdir(root_path+'/Results_general_algorithm')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'yes' not in filename and 'classes' not in filename
    and 'averaged' not in filename):
        total_name_list.append(filename)


###########################################################################################
##########                              DATASET LEVEL                          ############
###########################################################################################


data_list = ['ionosphere','mammographic','WineQualityRed_5vs6']

# path_to_save = root_path+'/Analysis_results_ranking_avg'

# data_i = 'Data1_'

for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    data_total = pd.DataFrame()
    for file in list_match:
        print(list_match)
        os.chdir(root_path + '/Results_general_algorithm')
        name = file[25:]
        data = pd.read_csv(file)
        if ('split6_alpha8' in name):
            data.columns = [str(col) + '_split6_alpha8' for col in data.columns]
        elif ('split6_alpha16' in name):
            data.columns = [str(col) + '_split6_alpha16' for col in data.columns]
        elif ('split8_alpha8' in name):
            data.columns = [str(col) + '_split8_alpha8' for col in data.columns]
        elif ('split8_alpha16' in name):
            data.columns = [str(col) + '_split8_alpha16' for col in data.columns]
        elif ('split10_alpha8' in name):
            data.columns = [str(col) + '_split10_alpha8' for col in data.columns]
        elif ('split10_alpha16' in name):
            data.columns = [str(col) + '_split10_alpha16' for col in data.columns]
        data.columns.values[0:2] = ['n_ensemble', 'weights']
        data_total = pd.concat([data_total, data], axis=1)
    data_total = data_total.loc[:, ~data_total.columns.duplicated()]  # remove duplicate columns
    # data_total.columns

    df_long_host = pd.melt(data_total[data_total['weights'] == 'Hostility'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_Hostility_dataset_mean_split6_alpha8', 'Boots_Hostility_dataset_mean_split6_alpha16',
                                  'Boots_Hostility_dataset_mean_split8_alpha8', 'Boots_Hostility_dataset_mean_split8_alpha16',
                                  'Boots_Hostility_dataset_mean_split10_alpha8', 'Boots_Hostility_dataset_mean_split10_alpha16'],
                      value_name='Complexity')



    # data_aux_host = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_Hostility_dataset_mean_split6_alpha8']]
    # data_aux_host.columns = ['n_ensemble','weights','Complexity']
    # data_aux_host['variable'] = 'Uniform'

    # df_long_host = pd.concat([df_long_host,data_aux_host],axis=0)



    # Boxplot
    ax = sns.boxplot(y=df_long_host["Complexity"], x=df_long_host["variable"],
                order=['Boots_Hostility_dataset_mean_split6_alpha8', 'Boots_Hostility_dataset_mean_split6_alpha16',
                                  'Boots_Hostility_dataset_mean_split8_alpha8', 'Boots_Hostility_dataset_mean_split8_alpha16',
                                  'Boots_Hostility_dataset_mean_split10_alpha8', 'Boots_Hostility_dataset_mean_split10_alpha16'],
                     color='white')
    sns.stripplot(data=df_long_host, x="variable", y="Complexity", dodge=True, ax=ax,
                  order=['Boots_Hostility_dataset_mean_split6_alpha8', 'Boots_Hostility_dataset_mean_split6_alpha16',
                                  'Boots_Hostility_dataset_mean_split8_alpha8', 'Boots_Hostility_dataset_mean_split8_alpha16',
                                  'Boots_Hostility_dataset_mean_split10_alpha8', 'Boots_Hostility_dataset_mean_split10_alpha16'])
    ax.set_xticklabels(['s6_a8', 's6_a16', 's8_a8','s8_a16',
                        's10_a8','s10_a16'],
                       rotation=40)
    plt.title(data_i)
    plt.show()


