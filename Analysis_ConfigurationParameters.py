## In this script we read, aggregate/summarize the results of ComplexityDrivenBagging
## for the different parameters tested in order to determine the best configuration of parameters
## for our method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

root_path = os.getcwd()


##########################################################################
########              SUMMARIZE OF ALL AGGREGATED CSVs            ########
##########################################################################

# path_csv = os.chdir(root_path+'/Results_general_algorithm')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and filename.startswith('Aggregated'):
#         total_name_list.append(filename)
#
# # len(total_name_list) # 7520
#
# # General df to save all the results
# cols = ['weights','accuracy_mean_mean','accuracy_mean_median', 'accuracy_mean_std','Dataset','alpha','split']
# df_total = pd.DataFrame(columns=cols)
# i = 0
# for data_file in total_name_list:
#     i = i + 1
#     # print(data_file)
#     print(i)
#     file = data_file
#     name_data = data_file[data_file.find('AggregatedResults_CDB_') + len('AggregatedResults_CDB_'):data_file.rfind('_split')]
#     alpha = data_file[data_file.find('alpha') + len('alpha'):data_file.rfind('.csv')]
#     split = data_file[data_file.find('split') + len('split'):data_file.rfind('_alpha')]
#     data = pd.read_csv(file)
#     df_summary = data.groupby(by='weights', as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
#     df_summary.columns = ['weights','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
#     df_summary['Dataset'] = name_data
#     df_summary['alpha'] = alpha
#     df_summary['split'] = split
#
#     df_total = pd.concat([df_total,df_summary])
#
# # df_total.shape # 67680
#
# # Reorder columns
# df_total = df_total.reindex(columns=['Dataset', 'alpha', 'split','weights', 'accuracy_mean_mean', 'accuracy_mean_median',
#        'accuracy_mean_std'])
# # To save the results
# path_to_save = root_path+'/Results_general_algorithm'
# os.chdir(path_to_save)
# nombre_csv_agg = 'SummarizeResults_ParameterConfiguration_CDB.csv'
# df_total.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)


##########################################################################
########                      SUMMARIZED CSV                      ########
##########################################################################
path_csv = os.chdir(root_path+'/Results_general_algorithm')
df_total = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB.csv')

### Heatmap per complexity measure
df_total_host = df_total.loc[df_total['weights'] == 'Hostility',:]
summary_host = df_total_host.groupby(['alpha','split'], as_index=False)['accuracy_mean_mean'].mean()
summary_host = pd.DataFrame(summary_host)

df_to_plot = summary_host.pivot(index='alpha', columns='split', values='accuracy_mean_mean')
df_to_plot.sort_index(level=0, inplace=True, ascending=False)

fig, ax = plt.subplots(figsize=(14,6))
p1 = sns.heatmap(df_to_plot, cmap="YlGnBu", annot=True)
plt.show()






