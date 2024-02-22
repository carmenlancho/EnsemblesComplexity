##############################################################################################################
##### SCRIPT TO COMPARE THE PERFORMANCE OF ALL METHODS BY ANALYZING WIN TIE LOSS AND STATISTICAL TESTS #####
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
#################################
### COMPLEXITY-DRIVEN BAGGING: FILTRO DE PARÁMETROS ###
#################################
path_csv = os.chdir(root_path+'/Results_general_algorithm')
# df_total_to_filter = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB.csv')
df_total_to_filter = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB_from20ensembles.csv')
# df_total_to_filter = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB_from30ensembles.csv')
# este archivo ha sido previamente agregado en el script Analysis_ConfigurationParameters
df_total_to_filter.shape

# # Aquí filtro los parámetros que no queremos
# # Comenzamos por alpha = [2,4,6,8,10] y s = [1,2,4,6,8,10,12,14,16,18,20] ## Filter1Params
# values_alpha = [12,14,16,18,20]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
# values_split = [22,24,26,28,30]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]
#
# #  Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,6,8,10] ## Filter2Params
# values_alpha = [12,14,16,18,20]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
# values_split = [12,14,16,18,20,22,24,26,28,30]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]
#
# # Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,8,12,16,20] ## Filter3Params
# values_alpha = [12,14,16,18,20]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
# values_split = [22,24,26,28,30,6,10,14,18]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]
#
# # Seguimos con alpha = [2,6,10] y s = [1,2,4,8,14,20] ## Filter4Params
# values_alpha = [12,14,16,18,20,4,8]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
# values_split = [22,24,26,28,30,6,10,12,16,18]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]
#
# #  Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,6,8,10,12,14] ## Filter5Params
# values_alpha = [12,14,16,18,20]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
# values_split = [16,18,20,22,24,26,28,30]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

#  Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,6,8] ## Filter6Params
values_alpha = [12,14,16,18,20]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [10,12,14,16,18,20,22,24,26,28,30]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

# #  Seguimos con alpha = [2,6,10] y s = [1,2,4,8,12,14] ## Filter7Params
# values_alpha = [12,14,16,18,20,4,8]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
# values_split = [6,10,16,18,20,22,24,26,28,30]
# df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

df_total = df_total_to_filter




################################
##      STANDARD BAGGING     ###
################################
path_csv = os.chdir(root_path+'/Results_StandardBagging')
# df_standard = pd.read_csv('SummarizeResults_StandardBagging.csv') ## all n_ensembles
df_standard = pd.read_csv('SummarizeResults_StandardBagging_from20ensembles.csv') # from20ensembles
# df_standard = pd.read_csv('SummarizeResults_StandardBagging_from30ensembles.csv') # from 30 ensembles


################################
##       MIXED BAGGING       ###
################################

path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')
# df_mixed = pd.read_csv('SummarizeResults_MixedBagging.csv') # all n_trees
df_mixed = pd.read_csv('SummarizeResults_MixedBagging_from20ensembles.csv') # from 20 n_trees
# df_mixed = pd.read_csv('SummarizeResults_MixedBagging_from30ensembles.csv') # from 30 n_trees


#########################################################################################################
#####                                SUMMARY TABLE OF ALL MODELS                                    #####
#########################################################################################################


table_comparison = pd.DataFrame(columns=['Dataset','Standard_Bag_mean','Standard_Bag_std'])
table_comparison['Dataset'] = df_standard['Dataset']
table_comparison['Standard_Bag_mean'] = df_standard["accuracy_mean_mean"]
table_comparison['Standard_Bag_std'] = df_standard["accuracy_mean_std"]
table_comparison.set_index('Dataset', inplace=True)

## Grouped and incremented
df_mixed_aux = df_mixed.loc[df_mixed['model']=='Grouped_Mixed_Bagging',
                                    ['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_mixed_aux.columns = ['Dataset', 'Grouped_accuracy_mean_mean', 'Grouped_accuracy_mean_std']
df_mixed_inc = df_mixed.loc[df_mixed['model']=='Incremental_Mixed_Bagging',
                                    ['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_mixed_inc.columns = ['Dataset', 'Incremental_accuracy_mean_mean', 'Incremental_accuracy_mean_std']
df_mixed_aux.set_index('Dataset', inplace=True)
df_mixed_inc.set_index('Dataset', inplace=True)

table_comparison = table_comparison.join(df_mixed_aux[['Grouped_accuracy_mean_mean','Grouped_accuracy_mean_std']])
table_comparison = table_comparison.join(df_mixed_inc[['Incremental_accuracy_mean_mean','Incremental_accuracy_mean_std']])
table_comparison.columns = ['Standard_Bag_mean', 'Standard_Bag_std', 'Grouped_Bag_mean', 'Grouped_Bag_std',
                            'Incre_Bag_mean', 'Incre_Bag_std']





## Best parameters per dataset in CDB
best_param = df_total.loc[df_total.groupby(["Dataset", "weights"])["accuracy_mean_mean"].idxmax()]

# Hostility
df_aux = best_param.loc[best_param['weights']=='Hostility',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_Host_mean',
                                   'accuracy_mean_std':'CDB_Host_std'}, inplace = True)

# kDN
df_aux = best_param.loc[best_param['weights']=='kDN',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_kDN_mean',
                                   'accuracy_mean_std':'CDB_kDN_std'}, inplace = True)

# CLD
df_aux = best_param.loc[best_param['weights']=='CLD',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_CLD_mean',
                                   'accuracy_mean_std':'CDB_CLD_std'}, inplace = True)

# LSC
df_aux = best_param.loc[best_param['weights']=='LSC',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_LSC_mean',
                                   'accuracy_mean_std':'CDB_LSC_std'}, inplace = True)

# N1
df_aux = best_param.loc[best_param['weights']=='N1',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_N1_mean',
                                   'accuracy_mean_std':'CDB_N1_std'}, inplace = True)

# N2
df_aux = best_param.loc[best_param['weights']=='N2',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_N2_mean',
                                   'accuracy_mean_std':'CDB_N2_std'}, inplace = True)

# DCP
df_aux = best_param.loc[best_param['weights']=='DCP',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_DCP_mean',
                                   'accuracy_mean_std':'CDB_DCP_std'}, inplace = True)

# TD_U
df_aux = best_param.loc[best_param['weights']=='TD_U',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_TD_U_mean',
                                   'accuracy_mean_std':'CDB_TD_U_std'}, inplace = True)

# F1
df_aux = best_param.loc[best_param['weights']=='F1',['Dataset','accuracy_mean_mean','accuracy_mean_std']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux[['accuracy_mean_mean','accuracy_mean_std']])
table_comparison.rename(columns = {'accuracy_mean_mean':'CDB_F1_mean',
                                   'accuracy_mean_std':'CDB_F1_std'}, inplace = True)



# Deletion of multiclass problems (segment, cleveland and analcatdata_authorship)
table_comparison.drop(['analcatdata_authorship', 'cleveland', 'segment'], inplace=True)



