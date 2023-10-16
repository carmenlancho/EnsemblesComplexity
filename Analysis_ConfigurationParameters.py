## In this script we read, aggregate/summarize the results of ComplexityDrivenBagging
## for the different parameters tested in order to determine the best configuration of parameters
## for our method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

root_path = os.getcwd()


path_csv = os.chdir(root_path+'/Results_general_algorithm')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('Aggregated'):
        total_name_list.append(filename)

# General df to save all the results
cols = ['weights','accuracy_mean_mean','accuracy_mean_median', 'accuracy_mean_std','Dataset','alpha','split']
df_total = pd.DataFrame(columns=cols)

for data_file in total_name_list:
    print(data_file)
    file = data_file
    name_data = data_file[data_file.find('AggregatedResults_CDB_') + len('AggregatedResults_CDB_'):data_file.rfind('_split')]
    alpha = data_file[data_file.find('alpha') + len('alpha'):data_file.rfind('.csv')]
    split = data_file[data_file.find('split') + len('split'):data_file.rfind('_alpha')]
    data = pd.read_csv(file)
    df_summary = data.groupby(by='weights', as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
    df_summary.columns = ['weights','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
    df_summary['Dataset'] = name_data
    df_summary['alpha'] = alpha
    df_summary['split'] = split

    df_total = pd.concat([df_total,df_summary])

# Reorder columns
df_total = df_total.reindex(columns=['Dataset', 'alpha', 'split','weights', 'accuracy_mean_mean', 'accuracy_mean_median',
       'accuracy_mean_std'])
# To save the results
path_to_save = root_path+'/Results_general_algorithm'
os.chdir(path_to_save)
nombre_csv_agg = 'SummarizeResults_ParameterConfiguration_CDB.csv'
df_total.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)




