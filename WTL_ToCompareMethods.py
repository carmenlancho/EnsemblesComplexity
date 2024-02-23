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

table_comparison_mean = table_comparison.loc[:,table_comparison.columns[table_comparison.columns.str.endswith('mean')]]

methods = ['Standard_Bag_mean','Grouped_Bag_mean','Incre_Bag_mean','CDB_Host_mean',
           'CDB_kDN_mean','CDB_CLD_mean','CDB_LSC_mean','CDB_N1_mean','CDB_N2_mean',
           'CDB_DCP_mean','CDB_TD_U_mean','CDB_F1_mean']
method = 'CDB_Host_mean'
df_mean = table_comparison_mean
def rank_df(df_mean, method, methods=methods):
    rank_index = [i for i in methods if i != method]
    wtl_df = pd.DataFrame(index=rank_index,
                          columns=['Wins', 'Ties', 'Losses'])
    for i in rank_index:
        wtl_df.loc[i, 'Wins'] = sum(df_mean[method] > df_mean[i])
        wtl_df.loc[i, 'Losses'] = sum(df_mean[method] < df_mean[i])
        wtl_df.loc[i, 'Ties'] = len(df_mean) - wtl_df.loc[i, 'Wins'] - wtl_df.loc[i, 'Losses']
    wtl_plot = wtl_df.copy()
    wtl_plot['Ties'] = wtl_df['Wins'] + wtl_df['Ties']
    wtl_plot['Losses'] = wtl_plot['Ties'] + wtl_df['Losses']

    return wtl_df, wtl_plot

total_dict = {}
for method_i in methods:
    wtl_df,_ = rank_df(table_comparison_mean, method_i, methods=methods)
    total_dict[method_i] = wtl_df


#### Hacemos lo mismo pero con distintos n_ensemble: 20, 50, 100, 150, 200

##########################################################################
########              RESULTS FROM CDB WITH DIFFERENT TREES            ########
##########################################################################
# el tamaño de ensamblado que queremos
# el mínimo es 20 por el tema de cubrir el espacio de complejidad en función del parámetro split
# desired_ntrees = [19,29,49,99,149,199]
#
# path_csv = os.chdir(root_path+'/Results_general_algorithm/Aggregated_results')
# # Extraemos los nombres de todos los ficheros
# # Filtramos en función de los parámetros que queremos
# alphas_v= ['alpha2','alpha4','alpha6','alpha8','alpha10']
# split_v= ['split1','split2','split4','split6','split8']
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and filename.startswith('Aggregated') and (any(alpha_text in filename for alpha_text in alphas_v)) and (any(split_text in filename for split_text in split_v)):
#         total_name_list.append(filename)
#
# len(total_name_list)
#
# # General df to save all the results
# cols = ['n_ensemble','weights','accuracy_mean', 'accuracy_std','Dataset','alpha','split']
# df_total = pd.DataFrame(columns=cols)
# # i=0
# for data_file in total_name_list:
#     # i = i+1
#     # print(i)
#     # print(data_file)
#     file = data_file
#     name_data = data_file[data_file.find('AggregatedResults_CDB_') + len('AggregatedResults_CDB_'):data_file.rfind('_split')]
#     alpha = data_file[data_file.find('alpha') + len('alpha'):data_file.rfind('.csv')]
#     split = data_file[data_file.find('split') + len('split'):data_file.rfind('_alpha')]
#     data = pd.read_csv(file)
#     # data.shape
#     # # Solo queremos los n_ensembles 20,30,50,100,150,200
#     data = data[data['n_ensemble'].isin(desired_ntrees)] # from 20 ensembles
#     # Selection of columnas
#     data_filter = data[['n_ensemble','weights','accuracy_mean', 'accuracy_std']]
#     data_filter.insert(4, "Dataset", name_data)
#     data_filter.insert(5, "alpha", alpha)
#     data_filter.insert(6, "split", split)
#
#     df_total = pd.concat([df_total,data_filter])
#
# # df_total.shape
# df_total.columns
# # Reorder columns
# df_total = df_total.reindex(columns=['Dataset','n_ensemble', 'alpha', 'split','weights', 'accuracy_mean', 'accuracy_std'])
# df_total.reset_index(drop=True,inplace=True)
# # To save the results
# path_to_save = root_path+'/Results_general_algorithm'
# os.chdir(path_to_save)
# nombre_csv = 'Results_CDB_Filter6Parameters_20_30_50_100_150_200_ensembles.csv'
# df_total.to_csv(nombre_csv, encoding='utf_8_sig', index=False)






################################
##      STANDARD BAGGING     ###
################################
# path_csv = os.chdir(root_path+'/Results_StandardBagging')
#
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and filename.startswith('Aggregated'):
#         total_name_list.append(filename)
# # len(total_name_list)
#
#
# desired_ntrees = [19,29,49,99,149,199]
#
# # General df to save all the results
# cols = ['n_ensemble','weights','accuracy_mean', 'accuracy_std','Dataset']
# df_standard = pd.DataFrame(columns=cols)
# for data_file in total_name_list:
#     # print(data_file)
#     file = data_file
#     name_data = data_file[data_file.find('AggregatedResults_StandardBagging_') +
#                           len('AggregatedResults_StandardBagging_'):data_file.rfind('.csv')]
#     data = pd.read_csv(file)
#     # Solo queremos los n_ensembles 20,30,50,100,150,200
#     data = data[data['n_ensemble'].isin(desired_ntrees)] # from 20 ensembles
#     # Selection of columnas
#     data_filter = data[['n_ensemble','weights','accuracy_mean', 'accuracy_std']]
#     data_filter.insert(4, "Dataset", name_data)
#     df_standard = pd.concat([df_standard,data_filter])
#
# # To save the results
# path_to_save = root_path+'/Results_StandardBagging'
# os.chdir(path_to_save)
# nombre_csv = 'Results_StandardBagging_20_30_50_100_150_200_ensembles.csv.csv'
# df_standard.to_csv(nombre_csv, encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/Results_StandardBagging')
df_standard = pd.read_csv('Results_StandardBagging_20_30_50_100_150_200_ensembles.csv')





################################
##       MIXED BAGGING       ###
################################

# path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')
#
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and filename.startswith('Mixed_Bagging_aggregated'):
#         total_name_list.append(filename)
#
#
# desired_ntrees = [20,30,50,100,150,200]
#
# # General df to save all the results
# cols = ['n_trees','model','perf_value_mean', 'perf_value_std','Dataset']
# df_mixed = pd.DataFrame(columns=cols)
#
# for data_file in total_name_list:
#     print(data_file)
#     file = data_file
#     name_data = data_file[data_file.find('Mixed_Bagging_aggregated_') +
#                           len('Mixed_Bagging_aggregated_'):data_file.rfind('.csv')]
#     data = pd.read_csv(file)
#
#     # Solo queremos los n_ensembles 20,30,50,100,150,200
#     data = data[data['n_trees'].isin(desired_ntrees)] # from 20 ensembles
#     # For now, we only keep Grouped_Mixed_Bagging, Incremental_Mixed_Bagging and accuracy
#     models_v = ['Grouped_Mixed_Bagging', 'Incremental_Mixed_Bagging']
#     perf_v = ['acc']
#     data_filter = data[data['model'].isin(models_v) & data['perf_measure'].isin(perf_v)]
#     # Selection of columnas
#     data_filter = data_filter[['n_trees','model','perf_value_mean', 'perf_value_std']]
#     data_filter.insert(4, "Dataset", name_data)
#     df_mixed = pd.concat([df_mixed,data_filter])
#
# df_mixed.rename(columns={'perf_value_mean': 'accuracy_mean', 'perf_value_std': 'accuracy_std'}, inplace=True)
#
#
# # To save the results
# # path_to_save = os.chdir(root_path+'/MixedBagging/Adapted_results')
# # os.chdir(path_to_save)
# os.chdir(root_path+'/MixedBagging/Adapted_results')
# nombre_csv = 'Results_MixedBagging_20_30_50_100_150_200_ensembles.csv' # out_values2
# df_mixed.to_csv(nombre_csv, encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')
df_mixed = pd.read_csv('Results_MixedBagging_20_30_50_100_150_200_ensembles.csv') # from 20 n_trees





# Me quedo con el número de ensamblados que quiero y selecciono los mejores parámetros

def Results_n_ensembles(df_cdb, df_mixed, df_standard, n_trees):

    # Filter
    df_cdb_trees = df_cdb[df_cdb['n_ensemble'] == n_trees]
    df_standard_trees = df_standard[df_standard['n_ensemble'] == n_trees]
    df_mixed_trees = df_mixed[df_mixed['n_trees'] == (n_trees+1)]

    table_comparison = pd.DataFrame(columns=['Dataset', 'Standard_Bag_mean', 'Standard_Bag_std'])
    table_comparison['Dataset'] = df_standard_trees['Dataset']
    table_comparison['Standard_Bag_mean'] = df_standard_trees["accuracy_mean"]
    table_comparison['Standard_Bag_std'] = df_standard_trees["accuracy_std"]
    table_comparison.set_index('Dataset', inplace=True)

    ## Grouped and incremented
    df_mixed_aux = df_mixed_trees.loc[df_mixed_trees['model'] == 'Grouped_Mixed_Bagging',
    ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_mixed_aux.columns = ['Dataset', 'Grouped_accuracy_mean', 'Grouped_accuracy_std']
    df_mixed_inc = df_mixed_trees.loc[df_mixed_trees['model'] == 'Incremental_Mixed_Bagging',
    ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_mixed_inc.columns = ['Dataset', 'Incremental_accuracy_mean', 'Incremental_accuracy_std']
    df_mixed_aux.set_index('Dataset', inplace=True)
    df_mixed_inc.set_index('Dataset', inplace=True)

    table_comparison = table_comparison.join(df_mixed_aux[['Grouped_accuracy_mean', 'Grouped_accuracy_std']])
    table_comparison = table_comparison.join(
        df_mixed_inc[['Incremental_accuracy_mean', 'Incremental_accuracy_std']])
    table_comparison.columns = ['Standard_Bag_mean', 'Standard_Bag_std', 'Grouped_Bag_mean', 'Grouped_Bag_std',
                                'Incre_Bag_mean', 'Incre_Bag_std']

    ## Best parameters per dataset in CDB
    best_param = df_cdb_trees.loc[df_cdb_trees.groupby(["Dataset", "weights"])["accuracy_mean"].idxmax()]

    # Hostility
    df_aux = best_param.loc[
        best_param['weights'] == 'Hostility', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_Host_mean',
                                     'accuracy_std': 'CDB_Host_std'}, inplace=True)
    # kDN
    df_aux = best_param.loc[best_param['weights'] == 'kDN', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_kDN_mean',
                                     'accuracy_std': 'CDB_kDN_std'}, inplace=True)
    # CLD
    df_aux = best_param.loc[best_param['weights'] == 'CLD', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_CLD_mean',
                                     'accuracy_std': 'CDB_CLD_std'}, inplace=True)
    # LSC
    df_aux = best_param.loc[best_param['weights'] == 'LSC', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_LSC_mean',
                                     'accuracy_std': 'CDB_LSC_std'}, inplace=True)
    # N1
    df_aux = best_param.loc[best_param['weights'] == 'N1', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_N1_mean',
                                     'accuracy_std': 'CDB_N1_std'}, inplace=True)
    # N2
    df_aux = best_param.loc[best_param['weights'] == 'N2', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_N2_mean',
                                     'accuracy_std': 'CDB_N2_std'}, inplace=True)
    # DCP
    df_aux = best_param.loc[best_param['weights'] == 'DCP', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_DCP_mean',
                                     'accuracy_std': 'CDB_DCP_std'}, inplace=True)
    # TD_U
    df_aux = best_param.loc[best_param['weights'] == 'TD_U', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_TD_U_mean',
                                     'accuracy_std': 'CDB_TD_U_std'}, inplace=True)
    # F1
    df_aux = best_param.loc[best_param['weights'] == 'F1', ['Dataset', 'accuracy_mean', 'accuracy_std']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_F1_mean',
                                     'accuracy_std': 'CDB_F1_std'}, inplace=True)


    table_comparison_mean = table_comparison.loc[:,
                            table_comparison.columns[table_comparison.columns.str.endswith('mean')]]

    return table_comparison, table_comparison_mean


path_csv = os.chdir(root_path+'/Results_general_algorithm')
df_cdb = pd.read_csv('Results_CDB_Filter6Parameters_20_30_50_100_150_200_ensembles.csv')

n_trees = 29
_, table_comparison_n_trees = Results_n_ensembles(df_cdb, df_mixed, df_standard, n_trees)


#### WTL
total_dict_30 = {}
for method_i in methods:
    wtl_df,_ = rank_df(table_comparison_n_trees, method_i, methods=methods)
    total_dict_30[method_i] = wtl_df



