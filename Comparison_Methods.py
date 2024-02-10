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
### COMPLEXITY-DRIVEN BAGGING: FILTRO DE PARÁMETROS ###
#################################
path_csv = os.chdir(root_path+'/Results_general_algorithm')
# df_total_to_filter = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB.csv')
df_total_to_filter = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB_from20ensembles.csv')
# df_total_to_filter = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB_from30ensembles.csv')
# este archivo ha sido previamente agregado en el script Analysis_ConfigurationParameters
df_total_to_filter.shape

# Aquí filtro los parámetros que no queremos
# Comenzamos por alpha = [2,4,6,8,10] y s = [1,2,4,6,8,10,12,14,16,18,20] ## Filter1Params
values_alpha = [12,14,16,18,20]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [22,24,26,28,30]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

#  Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,6,8,10] ## Filter2Params
values_alpha = [12,14,16,18,20]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [12,14,16,18,20,22,24,26,28,30]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

# Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,8,12,16,20] ## Filter3Params
values_alpha = [12,14,16,18,20]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [22,24,26,28,30,6,10,14,18]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

# Seguimos con alpha = [2,6,10] y s = [1,2,4,8,14,20] ## Filter4Params
values_alpha = [12,14,16,18,20,4,8]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [22,24,26,28,30,6,10,12,16,18]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

#  Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,6,8,10,12,14] ## Filter5Params
values_alpha = [12,14,16,18,20]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [16,18,20,22,24,26,28,30]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

#  Seguimos con alpha = [2,4,6,8,10] y s = [1,2,4,6,8] ## Filter6Params
values_alpha = [12,14,16,18,20]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [10,12,14,16,18,20,22,24,26,28,30]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]

#  Seguimos con alpha = [2,6,10] y s = [1,2,4,8,12,14] ## Filter7Params
values_alpha = [12,14,16,18,20,4,8]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['alpha'].isin(values_alpha)]
values_split = [6,10,16,18,20,22,24,26,28,30]
df_total_to_filter = df_total_to_filter[~df_total_to_filter['split'].isin(values_split)]






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
# # Quitamos n_ensemble = 0 y 9, para empezar en 20
# # hacemos otro empezando en 30
# out_values1 = [0,9]
# out_values2 = [0,9,19]
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
#     # Quitamos los n_ensemble que no queremos
#     data = data[~data['n_ensemble'].isin(out_values1)] # from 20 ensembles
#     # data = data[~data['n_ensemble'].isin(out_values2)]  # from 30 ensembles
#     df_summary = data.groupby(by='weights', as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
#     df_summary.columns = ['weights','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
#     df_summary['Dataset'] = name_data
#
#     df_standard = pd.concat([df_standard,df_summary])
#
# # To save the results
# path_to_save = root_path+'/Results_StandardBagging'
# os.chdir(path_to_save)
# nombre_csv_agg = 'SummarizeResults_StandardBagging_from20ensembles.csv' # out_values1
# # nombre_csv_agg = 'SummarizeResults_StandardBagging_from30ensembles.csv' # out_values2
# df_standard.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/Results_StandardBagging')
# df_standard = pd.read_csv('SummarizeResults_StandardBagging.csv') ## all n_ensembles
df_standard = pd.read_csv('SummarizeResults_StandardBagging_from20ensembles.csv') # from20ensembles
# df_standard = pd.read_csv('SummarizeResults_StandardBagging_from30ensembles.csv') # from 30 ensembles


################################
##       MIXED BAGGING       ###
################################
#
# path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')
#
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and filename.startswith('Mixed_Bagging_aggregated'):
#         total_name_list.append(filename)
# # len
#
# # Quitamos n_ensemble = 0 y 9, para empezar en 20
# # hacemos otro empezando en 30
# out_values1 = [2,10]
# out_values2 = [2,10,20]
#
# # General df to save all the results
# cols = ['model','perf_measure','accuracy_mean_mean','accuracy_mean_median', 'accuracy_mean_std','Dataset']
# df_mixed = pd.DataFrame(columns=cols)
#
# for data_file in total_name_list:
#     print(data_file)
#     file = data_file
#     name_data = data_file[data_file.find('Mixed_Bagging_aggregated_') +
#                           len('Mixed_Bagging_aggregated_'):data_file.rfind('.csv')]
#     data = pd.read_csv(file)
#     # Quitamos los n_trees que no queremos
#     # data = data[~data['n_trees'].isin(out_values1)] # from 20 ensembles
#     data = data[~data['n_trees'].isin(out_values2)]  # from 30 ensembles
#     df_summary = data.groupby(by=['model','perf_measure'], as_index=False).agg({'perf_value_mean': [np.mean, np.median, np.std]})
#     # data.groupby(by=['perf_measure','model'], as_index=False).agg({'n_trees': 'median'})
#     # aa = data.loc[(data['model'] == 'AdaBoost') & (data['perf_measure'] == 'acc')]
#     # aa['perf_value_mean'].std()
#
#     df_summary.columns = ['model','perf_measure','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
#     df_summary['Dataset'] = name_data
#     # For now, we only keep Grouped_Mixed_Bagging, Incremental_Mixed_Bagging and accuracy
#     models_v = ['Grouped_Mixed_Bagging', 'Incremental_Mixed_Bagging']
#     perf_v = ['acc']
#     df_summary = df_summary[df_summary['model'].isin(models_v) & df_summary['perf_measure'].isin(perf_v)]
#
#     df_mixed = pd.concat([df_mixed,df_summary])
#
# # To save the results
# # path_to_save = os.chdir(root_path+'/MixedBagging/Adapted_results')
# # os.chdir(path_to_save)
# os.chdir(root_path+'/MixedBagging/Adapted_results')
# # nombre_csv_agg = 'SummarizeResults_MixedBagging.csv'
# # nombre_csv_agg = 'SummarizeResults_MixedBagging_from20ensembles.csv' # out_values1
# nombre_csv_agg = 'SummarizeResults_MixedBagging_from30ensembles.csv' # out_values2
# df_mixed.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')
# df_mixed = pd.read_csv('SummarizeResults_MixedBagging.csv') # all n_trees
df_mixed = pd.read_csv('SummarizeResults_MixedBagging_from20ensembles.csv') # from 20 n_trees
# df_mixed = pd.read_csv('SummarizeResults_MixedBagging_from30ensembles.csv') # from 30 n_trees



#########################################################################################################
#####                                SUMMARY TABLE OF ALL MODELS                                    #####
#########################################################################################################
# por el filtro
df_total = df_total_to_filter

table_comparison = pd.DataFrame(columns=['Dataset','Standard_Bag'])
table_comparison['Dataset'] = df_standard['Dataset']
df_standard["Table"] = (round(df_standard["accuracy_mean_mean"],3).astype('str') +
                       " (" +
                        round(df_standard["accuracy_mean_std"],3).astype('str') + ")")
table_comparison['Standard_Bag'] = df_standard["Table"]
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

df_mixed_aux['Grouped_Bag'] = (round(df_mixed_aux["Grouped_accuracy_mean_mean"],3).astype('str') +
                       " (" +
                        round(df_mixed_aux["Grouped_accuracy_mean_std"],3).astype('str') + ")")

df_mixed_inc['Incre_Bag'] = (round(df_mixed_inc["Incremental_accuracy_mean_mean"],3).astype('str') +
                       " (" +
                        round(df_mixed_inc["Incremental_accuracy_mean_std"],3).astype('str') + ")")


table_comparison = table_comparison.join(df_mixed_aux['Grouped_Bag'])
table_comparison = table_comparison.join(df_mixed_inc['Incre_Bag'])


## Best parameters per dataset in CDB
best_param = df_total.loc[df_total.groupby(["Dataset", "weights"])["accuracy_mean_mean"].idxmax()]
best_param['Table'] = (round(best_param["accuracy_mean_mean"],3).astype('str') +
                       " (" +
                        round(best_param["accuracy_mean_std"],3).astype('str') + ")")

df_aux = best_param.loc[best_param['weights']=='Hostility',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_Host'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='kDN',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_kDN'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='CLD',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_CLD'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='LSC',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_LSC'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='N1',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_N1'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='N2',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_N2'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='DCP',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_DCP'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='TD_U',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_TD_U'}, inplace = True)

df_aux = best_param.loc[best_param['weights']=='F1',['Dataset','Table']]
df_aux.set_index('Dataset', inplace=True)
table_comparison = table_comparison.join(df_aux['Table'])
table_comparison.rename(columns = {'Table':'CDB_Best_F1'}, inplace = True)

# To save the results
# path_to_save = os.chdir(root_path+'/Results_Comparison_Methods')
# os.chdir(path_to_save)
os.chdir(root_path+'/Results_Comparison_Methods')
nombre_csv_agg = 'table_comparison_from30_Filter7Params.csv'
table_comparison.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=True)
table_comparison

## Parameters
best_param['param'] = ('a=' + best_param["alpha"].astype('str') +
                       ', s=' + best_param["split"].astype('str'))
parameters_CM = best_param.pivot(index='Dataset', columns='weights', values='param')
nombre_csv_agg_param = 'parameters_CM_from30_Filter7Params.csv'
parameters_CM.to_csv(nombre_csv_agg_param, encoding='utf_8_sig', index=True)
nombre_csv_agg_param2 = 'best_param_from20_Filter6Params.csv'
best_param.to_csv(nombre_csv_agg_param2, encoding='utf_8_sig', index=True)



 # FALTA ORDENAR POR COMPLEXITY



#########################################################################################################
#####                                  PLOT OF THE EVOLUTION                                        #####
#########################################################################################################

###### Read and plot standard bagging ####################################################
path_csv = os.chdir(root_path+'/Results_StandardBagging')

# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('Aggregated'):
        total_name_list.append(filename)
# len(total_name_list)


# General df to save all the results
cols = ['n_ensemble','StandardBagging_accuracy','Dataset']
df_to_plot_standard = pd.DataFrame(columns=cols)
for data_file in total_name_list:
    # print(data_file)
    file = data_file
    name_data = data_file[data_file.find('AggregatedResults_StandardBagging_') +
                          len('AggregatedResults_StandardBagging_'):data_file.rfind('.csv')]
    data_df = pd.read_csv(file)
    data_df2 = data_df.iloc[:,[0,2]].copy()
    data_df2['Dataset'] = name_data
    data_df2.columns = cols
    df_to_plot_standard = pd.concat([df_to_plot_standard, data_df2])

# df_to_plot_standard['n_ensemble'] = df_to_plot_standard['n_ensemble'] + 1
# df_to_plot_standard[df_to_plot_standard['n_ensemble'] ==1] = 0

# # Plot
# dataset = 'profb'
# data_plot = df_to_plot.loc[(df_to_plot['Dataset'] == dataset) & (df_to_plot['n_ensemble']>10)]
# plt.plot(data_plot['n_ensemble'],data_plot['StandardBagging_accuracy'],'o', linestyle = 'dotted',label = 'StandardBagging')
# plt.show()

###### Read and plot Mixed and incremental bagging ####################################################


path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')

# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('Mixed_Bagging_aggregated'):
        total_name_list.append(filename)

cols = ['n_ensemble','GroupedBagging_accuracy','Dataset','IncrementalBagging_accuracy']
df_to_plot_mixed = pd.DataFrame(columns=cols)

for data_file in total_name_list:
    print(data_file)
    file = data_file
    name_data = data_file[data_file.find('Mixed_Bagging_aggregated_') +
                          len('Mixed_Bagging_aggregated_'):data_file.rfind('.csv')]
    data_df = pd.read_csv(file)
    data_df2 = data_df.loc[(data_df['model']=='Grouped_Mixed_Bagging') & (data_df['perf_measure']=='acc'), ['n_trees','perf_value_mean']]
    data_df2['Dataset'] = name_data
    data_df2.reset_index(drop=True, inplace=True)
    var = data_df.loc[
        (data_df['model'] == 'Incremental_Mixed_Bagging') & (data_df['perf_measure'] == 'acc'), ['perf_value_mean']]
    var.reset_index(drop=True, inplace=True)
    data_df2['Incremental_Mixed_Bagging'] = var
    data_df2.columns = cols
    df_to_plot_mixed = pd.concat([df_to_plot_mixed, data_df2])


# Plot
dataset = 'profb'
data_plot = df_to_plot_mixed.loc[(df_to_plot_mixed['Dataset'] == dataset) & (df_to_plot_mixed['n_ensemble']>10)]
plt.plot(data_plot['n_ensemble'],data_plot['GroupedBagging_accuracy'],'o', linestyle = 'dotted',
         label = 'Grouped_Mixed_Bagging',color = 'k')
plt.plot(data_plot['n_ensemble'],data_plot['IncrementalBagging_accuracy'],'o', linestyle = 'dashed',
         label = 'Incremental_Mixed_Bagging',color='orange')
plt.legend()
plt.show()





list_datasets = ['WineQualityRed_5vs6', 'Yeast_CYTvsNUC',
       'arrhythmia_cfs', 'bands', 'banknote_authentication', 'bodyfat',
       'breast-w', 'breastcancer', 'bupa', 'chatfield_4',
       'contraceptive_LS', 'contraceptive_NL', 'contraceptive_NS',
       'credit-g', 'diabetes', 'diabetic_retinopathy', 'fri_c0_250_50',
       'glass0', 'glass1', 'hill_valley_without_noise_traintest', 'ilpd',
       'ionosphere', 'liver-disorders', 'mammographic', 'musk1',
       'parkinsons', 'phoneme', 'pima', 'profb', 'ring', 'saheart',
        'sonar', 'spambase', 'spectfheart',
       'steel-plates-fault', 'sylvine', 'teaching_assistant_LH',
       'teaching_assistant_LM', 'teaching_assistant_MH', 'titanic',
       'vehicle2', 'vertebral_column', 'wdbc', 'yeast1']
len(list_datasets)

############## Complexity driven Bagging

path_csv = os.chdir(root_path+'/Results_general_algorithm/Aggregated_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('Aggregated'):
        total_name_list.append(filename)

path_csv = os.chdir(root_path+'/Results_Comparison_Methods')

best_param = pd.read_csv('best_param_from20_Filter6Params.csv')
best_param_host = best_param.loc[best_param['weights'] == 'Hostility',['Dataset','alpha','split']]

path_csv = os.chdir(root_path+'/Results_general_algorithm/Aggregated_results')
# General df to save all the results
cols = ['n_ensemble','CDB_Host_accuracy','Dataset']
df_to_plot_cdb = pd.DataFrame(columns=cols)
for dataset_i in list_datasets:
    print(dataset_i)
    # file = data_file
    alpha = best_param_host.loc[best_param_host['Dataset']==dataset_i,['alpha']].values[0]
    split = best_param_host.loc[best_param_host['Dataset'] == dataset_i, ['split']].values[0]
    param_to_search = dataset_i+ '_split'+str(split[0]) +'_alpha'+str(alpha[0])

    matching_file = [s for s in total_name_list if param_to_search in s][0]

    data_df = pd.read_csv(matching_file)
    data_df2 = data_df.loc[data_df['weights'] == 'Hostility', ['n_ensemble', 'accuracy_mean']].copy()
    data_df2['Dataset'] = dataset_i
    data_df2.columns = cols
    df_to_plot_cdb = pd.concat([df_to_plot_cdb, data_df2])


df_to_plot_cdb.reset_index(drop=True,inplace=True)


#### Plot

def plot_acc_ensembles(dataset_name, path_to_save):

    data_plot_standard = df_to_plot_standard.loc[(df_to_plot_standard['Dataset'] == dataset_name) & (df_to_plot_standard['n_ensemble']>10)]
    data_plot_cbd = df_to_plot_cdb.loc[(df_to_plot_cdb['Dataset'] == dataset_name) & (df_to_plot_cdb['n_ensemble']>10)]
    df_plot_mixed = df_to_plot_mixed.loc[(df_to_plot_mixed['Dataset'] == dataset_name) & (df_to_plot_mixed['n_ensemble']>10)]

    plt.plot(data_plot_standard['n_ensemble'],data_plot_standard['StandardBagging_accuracy'],'P', linestyle = 'dotted',
             label = 'Standard Bagging',color = 'k')
    plt.plot(data_plot_cbd['n_ensemble'],data_plot_cbd['CDB_Host_accuracy'],'o', linestyle = 'dashed',
             label = 'CDB_Hostility',color='orange')
    plt.plot(df_plot_mixed['n_ensemble'],df_plot_mixed['GroupedBagging_accuracy'],'^', linestyle = 'dotted',
             label = 'Grouped_Mixed_Bagging',color = 'c')
    plt.plot(df_plot_mixed['n_ensemble'],df_plot_mixed['IncrementalBagging_accuracy'],'s', linestyle = 'dashed',
             label = 'Incremental_Mixed_Bagging',color='red')
    plt.legend()
    plt.title(dataset_name)
    plt.tight_layout()
    # plt.show()
    os.chdir(path_to_save)
    plt.savefig('Accuracy_evolution_From20Ensembles_Host_' + str(dataset_name) + '.png')
    plt.clf()

    return


# dataset_name = 'profb'
path_to_save = root_path+'/Results_Comparison_Methods'

for dataset_name in list_datasets:
    plot_acc_ensembles(dataset_name, path_to_save)



