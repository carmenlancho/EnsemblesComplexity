##############################################################################################################
##### SCRIPT TO COMPARE THE PERFORMANCE OF COMPLEXITY-DRIVEN BAGGING, MIXED BAGGING AND STANDARD BAGGING #####
##############################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

root_path = os.getcwd()

#################################
### COMPLEXITY-DRIVEN BAGGING ###
#################################
path_csv = os.chdir(root_path+'/Results_general_algorithm')
df_total = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB.csv')
# este archivo ha sido previamente agregado en el script Analysis_ConfigurationParameters


#################################
###      STANDARD BAGGING     ###
#################################
# path_csv = os.chdir(root_path+'/Results_StandardBagging')
#
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and filename.startswith('Aggregated'):
#         total_name_list.append(filename)
# # len(total_name_list)
#
# # General df to save all the results
# cols = ['weights','accuracy_mean_mean','accuracy_mean_median', 'accuracy_mean_std','Dataset']
# df_standard = pd.DataFrame(columns=cols)
# i = 0
# for data_file in total_name_list:
#     i = i + 1
#     # print(data_file)
#     print(i)
#     file = data_file
#     name_data = data_file[data_file.find('AggregatedResults_StandardBagging_') +
#                           len('AggregatedResults_StandardBagging_'):data_file.rfind('.csv')]
#     data = pd.read_csv(file)
#     df_summary = data.groupby(by='weights', as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
#     df_summary.columns = ['weights','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
#     df_summary['Dataset'] = name_data
#
#     df_standard = pd.concat([df_standard,df_summary])
#
# # To save the results
# path_to_save = root_path+'/Results_StandardBagging'
# os.chdir(path_to_save)
# nombre_csv_agg = 'SummarizeResults_StandardBagging.csv'
# df_standard.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/Results_StandardBagging')
df_standard = pd.read_csv('SummarizeResults_StandardBagging.csv')


#################################
###       MIXED BAGGING       ###
#################################