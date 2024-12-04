## In this script, for the Bagging case with cycles, we compared all the methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


root_path = os.getcwd()



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
# # General df to save all the results
# cols = ['Dataset','weights','accuracy_mean', 'accuracy_std']
# df_standard = pd.DataFrame(columns=cols)
#
# for data_file in total_name_list:
#     # print(data_file)
#     file = data_file
#     name_data = data_file[data_file.find('AggregatedResults_StandardBagging_') +
#                           len('AggregatedResults_StandardBagging_'):data_file.rfind('.csv')]
#     data = pd.read_csv(file)
#     data['Dataset'] = name_data
#     df_standard = pd.concat([df_standard,data])
#
# # To save the results
# path_to_save = root_path+'/Results_StandardBagging'
# os.chdir(path_to_save)
# nombre_csv_agg = 'AllResults_StandardBagging.csv'
# # df_standard.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/Results_StandardBagging')
df_standard = pd.read_csv('AllResults_StandardBagging.csv') ## all n_ensembles

# Best results for standard bagging
best_param_standard = df_standard.loc[df_standard.groupby(["Dataset"])["accuracy_mean"].idxmax()]


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
# # len(total_name_list)
#
#
# # General df to save all the results
# cols = ['Dataset','n_trees','model','accuracy_mean','accuracy_std']
# df_mixed = pd.DataFrame(columns=cols)
#
# for data_file in total_name_list:
#     print(data_file)
#     file = data_file
#     name_data = data_file[data_file.find('Mixed_Bagging_aggregated_') +
#                           len('Mixed_Bagging_aggregated_'):data_file.rfind('.csv')]
#     data = pd.read_csv(file)
#     data_aux = data[['n_trees','model','perf_measure','perf_value_mean','perf_value_std']]
#     # For now, we only keep Grouped_Mixed_Bagging, Incremental_Mixed_Bagging and accuracy
#     data_aux = data_aux[data_aux['model'].isin(['Grouped_Mixed_Bagging','Incremental_Mixed_Bagging']) &
#                         (data_aux['perf_measure'] == 'acc')]
#
#     data_aux['Dataset'] = name_data
#     # Rename columns
#     data_aux.rename(columns={'perf_value_mean': 'accuracy_mean', 'perf_value_std': 'accuracy_std'}, inplace=True)
#     data_aux.drop('perf_measure', axis=1, inplace=True)
#
#     df_mixed = pd.concat([df_mixed,data_aux])

# To save the results
# os.chdir(root_path+'/MixedBagging/Adapted_results')
# df_mixed.to_csv('AllResults_MixedBagging.csv', encoding='utf_8_sig', index=False)

path_csv = os.chdir(root_path+'/MixedBagging/Adapted_results')
df_mixed = pd.read_csv('AllResults_MixedBagging.csv') # all n_trees

# Best result per dataset and complexity measure
best_param_mixed = df_mixed.loc[df_mixed.groupby(["Dataset", "model"])["accuracy_mean"].idxmax()]

############################################################################################
###########                            CDB with cycles                           ###########
############################################################################################

## We begin by loading the CDB results and the results regarding the maximum number of cycles
path_csv = os.chdir(root_path+'/Results_general_algorithm_cycles')
df_cdb = pd.read_csv('TotalAggregatedResults_ParameterConfiguration_CDB.csv')
df_cbd_info_cycles = pd.read_csv('CDB_cycles_ParametersComboAlphaSplit_dif_no_signif_cycles_mean_median_std_num_models.csv')
# max_num_cycles, max_num_models son las variables que indican el número máximo de modelos/ciclos a tener en cuenta
# si no queremos tener en cuenta aquellos para los que ya no hay diferencias significativas

# Valores que escogemos para alpha y split con el estudio de R CDB_cycles_AnalysisOfParameters
# Domain for split: [2, 4, 6, 8, 10, 12, 14]
# Domain for alpha: [2, 4, 6, 8, 10]
alpha_domain= [2, 4, 6, 8, 10]
split_domain = [2, 4, 6, 8, 10, 12, 14]

# For every dataset, we want to obtain, for every complexity measure, the accuracy of the best combination
# of alpha, split and number of cycles/ensemble
# We first filter the df following the final domain for split and alpha
df_cdb_f = df_cdb[df_cdb['alpha'].isin(alpha_domain) & df_cdb['split'].isin(split_domain)]



# Best result per dataset and complexity measure
best_param_cdb = df_cdb_f.loc[df_cdb_f.groupby(["Dataset", "weights"])["accuracy_mean"].idxmax()]


#############################################################################################
#############                         COMPARISON WTL                            #############
#############################################################################################
# lo ponemos junto en un df y comparamos con WTL el mejor resultado de cada uno


def Results_format_for_WTL(df_cdb_best, df_mixed_best, df_standard_best):

    table_comparison = pd.DataFrame(columns=['Dataset', 'Standard_Bag_mean', 'Standard_Bag_std','Standard_Bag_n_ensemble'])
    table_comparison['Dataset'] = df_standard_best['Dataset']
    table_comparison['Standard_Bag_mean'] = df_standard_best["accuracy_mean"]
    table_comparison['Standard_Bag_std'] = df_standard_best["accuracy_std"]
    table_comparison['Standard_Bag_n_ensemble'] = df_standard_best["n_ensemble"]
    table_comparison.set_index('Dataset', inplace=True)

    ## Grouped and incremented
    df_mixed_aux = df_mixed_best.loc[df_mixed_best['model'] == 'Grouped_Mixed_Bagging',
    ['Dataset', 'accuracy_mean', 'accuracy_std','n_trees']]
    df_mixed_aux.columns = ['Dataset', 'Grouped_accuracy_mean', 'Grouped_accuracy_std','Grouped_n_ensemble']
    df_mixed_inc = df_mixed_best.loc[df_mixed_best['model'] == 'Incremental_Mixed_Bagging',
    ['Dataset', 'accuracy_mean', 'accuracy_std','n_trees']]
    df_mixed_inc.columns = ['Dataset', 'Incremental_accuracy_mean', 'Incremental_accuracy_std','Incremental_n_ensemble']
    df_mixed_aux.set_index('Dataset', inplace=True)
    df_mixed_inc.set_index('Dataset', inplace=True)

    table_comparison = table_comparison.join(df_mixed_aux[['Grouped_accuracy_mean', 'Grouped_accuracy_std','Grouped_n_ensemble']])
    table_comparison = table_comparison.join(
        df_mixed_inc[['Incremental_accuracy_mean', 'Incremental_accuracy_std','Incremental_n_ensemble']])
    table_comparison.columns = ['Standard_Bag_mean', 'Standard_Bag_std','Standard_Bag_n_ensemble',
                                'Grouped_Bag_mean', 'Grouped_Bag_std','Grouped_n_ensemble',
                                'Incre_Bag_mean', 'Incre_Bag_std','Incremental_n_ensemble']

    ## Best parameters per dataset in CDB: df_cdb_best

    # Hostility
    df_aux = df_cdb_best.loc[
        df_cdb_best['weights'] == 'Hostility', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_Host_mean',
                                     'accuracy_std': 'CDB_Host_std',
                                     'n_ensemble':'CDB_Host_n_ensemble'}, inplace=True)
    # kDN
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'kDN', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_kDN_mean',
                                     'accuracy_std': 'CDB_kDN_std',
                                     'n_ensemble':'CDB_kDN_n_ensemble'}, inplace=True)
    # CLD
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'CLD', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_CLD_mean',
                                     'accuracy_std': 'CDB_CLD_std',
                                     'n_ensemble':'CDB_CLD_n_ensemble'}, inplace=True)
    # LSC
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'LSC', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_LSC_mean',
                                     'accuracy_std': 'CDB_LSC_std',
                                     'n_ensemble':'CDB_LSC_n_ensemble'}, inplace=True)
    # N1
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'N1', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_N1_mean',
                                     'accuracy_std': 'CDB_N1_std',
                                     'n_ensemble':'CDB_N1_n_ensemble'}, inplace=True)
    # N2
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'N2', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_N2_mean',
                                     'accuracy_std': 'CDB_N2_std',
                                     'n_ensemble':'CDB_N2_n_ensemble'}, inplace=True)
    # DCP
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'DCP', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_DCP_mean',
                                     'accuracy_std': 'CDB_DCP_std',
                                     'n_ensemble':'CDB_DCP_n_ensemble'}, inplace=True)
    # TD_U
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'TD_U', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_TD_U_mean',
                                     'accuracy_std': 'CDB_TD_U_std',
                                     'n_ensemble':'CDB_TD_U_n_ensemble'}, inplace=True)
    # F1
    df_aux = df_cdb_best.loc[df_cdb_best['weights'] == 'F1', ['Dataset', 'accuracy_mean', 'accuracy_std','n_ensemble']]
    df_aux.set_index('Dataset', inplace=True)
    table_comparison = table_comparison.join(df_aux[['accuracy_mean', 'accuracy_std','n_ensemble']])
    table_comparison.rename(columns={'accuracy_mean': 'CDB_F1_mean',
                                     'accuracy_std': 'CDB_F1_std',
                                     'n_ensemble':'CDB_F1_n_ensemble'}, inplace=True)


    table_comparison_mean = table_comparison.loc[:,
                            table_comparison.columns[table_comparison.columns.str.endswith('mean')]]

    return table_comparison, table_comparison_mean



table_comparison, table_comparison_mean = Results_format_for_WTL(best_param_cdb, best_param_mixed, best_param_standard)
# QUITAMOS RING PORQUE AUN NO ESTÁ
table_comparison.drop(['ring'],inplace=True)
table_comparison_mean.drop(['ring'],inplace=True)


methods = ['Standard_Bag_mean','Grouped_Bag_mean','Incre_Bag_mean','CDB_Host_mean',
           'CDB_kDN_mean','CDB_CLD_mean','CDB_LSC_mean','CDB_N1_mean','CDB_N2_mean',
           'CDB_DCP_mean','CDB_TD_U_mean','CDB_F1_mean']

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

method_i = 'CDB_Host_mean'
wtl_df, _ = rank_df(table_comparison_mean, method_i, methods=methods)
wtl_df_str = pd.DataFrame()
wtl_df_str['wtl'] = "(" + wtl_df.apply(lambda row: ','.join(row.values.astype(str)), axis=1) + ")"

# Estudio WTL comparando medias de accuracy
wtl_df_str = pd.DataFrame()
for method_i in methods[3:]:
    CM = method_i[len('CDB_'):-len('_mean')]
    wtl_df, _ = rank_df(table_comparison_mean, method_i, methods=methods)
    name_column = 'wtl_' + CM
    wtl_df_str[name_column] = "(" + wtl_df.apply(lambda row: ','.join(row.values.astype(str)), axis=1) + ")"
wtl_df_str.drop(['CDB_kDN_mean','CDB_CLD_mean','CDB_LSC_mean','CDB_N1_mean','CDB_N2_mean',
           'CDB_DCP_mean','CDB_TD_U_mean','CDB_F1_mean'],inplace=True)


## Hacemos estudio WTL para n_ensembles
table_comparison_n_ensemble = table_comparison.loc[:,
                        table_comparison.columns[table_comparison.columns.str.endswith('ensemble')]]
wtl_df_n_ensemble = pd.DataFrame()
methods = ['Standard_Bag_n_ensemble','Grouped_n_ensemble','Incremental_n_ensemble',
           'CDB_Host_n_ensemble',
           'CDB_kDN_n_ensemble','CDB_CLD_n_ensemble','CDB_LSC_n_ensemble',
           'CDB_N1_n_ensemble','CDB_N2_n_ensemble',
           'CDB_DCP_n_ensemble','CDB_TD_U_n_ensemble','CDB_F1_n_ensemble']
for method_i in methods[3:]:
    CM = method_i[len('CDB_'):-len('_n_ensemble')]
    wtl_df, _ = rank_df(table_comparison_n_ensemble, method_i, methods=methods)
    name_column = 'wtl_' + CM
    wtl_df_n_ensemble[name_column] = "(" + wtl_df.apply(lambda row: ','.join(row.values.astype(str)), axis=1) + ")"

# Here we prefer to loss, it means the number of ensembles is lower
wtl_df_n_ensemble.drop(['CDB_kDN_n_ensemble','CDB_CLD_n_ensemble','CDB_LSC_n_ensemble',
           'CDB_N1_n_ensemble','CDB_N2_n_ensemble',
           'CDB_DCP_n_ensemble','CDB_TD_U_n_ensemble','CDB_F1_n_ensemble'],inplace=True)

## Gráficos
table_comparison.columns


# Seleccionar datos para mean
mean_cols = ['Standard_Bag_mean',
       'Grouped_Bag_mean',  'Incre_Bag_mean',
       'CDB_Host_mean',  'CDB_kDN_mean', 'CDB_CLD_mean',
             'CDB_LSC_mean', 'CDB_N1_mean',
       'CDB_N2_mean',  'CDB_DCP_mean','CDB_TD_U_mean',
             'CDB_F1_mean']
mean_data = table_comparison[mean_cols]


# Reestructurar los datos para cada métrica
mean_data = table_comparison[[col for col in table_comparison.columns if '_mean' in col]].melt(var_name='Method', value_name='Mean')
std_data = table_comparison[[col for col in table_comparison.columns if '_std' in col]].melt(var_name='Method', value_name='STD')
ensemble_data = table_comparison[[col for col in table_comparison.columns if '_n_ensemble' in col]].melt(var_name='Method', value_name='Num ensemble')


sns.boxplot(x='Method', y='Mean', data=mean_data,color='#D4D7FC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


sns.boxplot(x='Method', y='STD', data=std_data,color='#D4D7FC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

sns.boxplot(x='Method', y='Num ensemble', data=ensemble_data,color='#D4D7FC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

######################################################################################################################
##########                       MULTIPLE COMPARISON SIGNIF                           ################################
######################################################################################################################

# We repeat the same analysis but only taking into account the number of cycles
# with significant differences
df_cdb_f.columns
path_csv = os.chdir(root_path+'/Results_general_algorithm_cycles')
# df con el máximo número de modelos con diferencia significativa
df_signif = pd.read_csv('CDB_cycles_ParametersComboAlphaSplit_dif_no_signif_cycles_mean_median_std_num_models.csv')

## Vamos a añadir a df_cdb_f la columna del máximo número de modelos

# Separamos la columna valores_combo en alpha y split
df_signif[['alpha', 'split']] = (
    df_signif['valores_combo']
    .str.extract(r'alpha(\d+)-split(\d+)')  # Extrae los números después de 'alpha' y 'split'
    .astype(int)  # Convierte a entero para asegurar el mismo tipo en ambos datasets
)

# print(df_cdb_f.dtypes)
# print(df_signif.dtypes)

# Merge entre df_cdb_f y df_signif
df_cdb_f['alpha'] = df_cdb_f['alpha'].astype(str)
df_cdb_f['split'] = df_cdb_f['split'].astype(str)
df_signif['alpha'] = df_signif['alpha'].astype(str)
df_signif['split'] = df_signif['split'].astype(str)
df_cdb_m = pd.merge(
    df_cdb_f,
    df_signif[['alpha', 'split', 'max_num_models']],  # Solo necesitamos estas columnas tras la separación
    on=['alpha', 'split'],
    how='left')

# Filtrar los valores donde n_ensemble es menor o igual al máximo permitido
df_cdb_signif = df_cdb_m[df_cdb_m['n_ensemble'] <= df_cdb_m['max_num_models']]

### Repetimos el análisis anterior pero solo con los que tienen diferencias significativas

# Best result per dataset and complexity measure
best_param_cdb_signif = df_cdb_signif.loc[df_cdb_signif.groupby(["Dataset", "weights"])["accuracy_mean"].idxmax()]


table_comparison, table_comparison_mean = Results_format_for_WTL(best_param_cdb_signif, best_param_mixed, best_param_standard)
# QUITAMOS RING PORQUE AUN NO ESTÁ
table_comparison.drop(['ring'],inplace=True)
table_comparison_mean.drop(['ring'],inplace=True)


# Estudio WTL comparando medias de accuracy
methods = ['Standard_Bag_mean','Grouped_Bag_mean','Incre_Bag_mean','CDB_Host_mean',
           'CDB_kDN_mean','CDB_CLD_mean','CDB_LSC_mean','CDB_N1_mean','CDB_N2_mean',
           'CDB_DCP_mean','CDB_TD_U_mean','CDB_F1_mean']
wtl_df_str_signif = pd.DataFrame()
for method_i in methods[3:]:
    CM = method_i[len('CDB_'):-len('_mean')]
    wtl_df, _ = rank_df(table_comparison_mean, method_i, methods=methods)
    name_column = 'wtl_' + CM
    wtl_df_str_signif[name_column] = "(" + wtl_df.apply(lambda row: ','.join(row.values.astype(str)), axis=1) + ")"
wtl_df_str_signif.drop(['CDB_kDN_mean','CDB_CLD_mean','CDB_LSC_mean','CDB_N1_mean','CDB_N2_mean',
           'CDB_DCP_mean','CDB_TD_U_mean','CDB_F1_mean'],inplace=True)



## Hacemos estudio WTL para n_ensembles
table_comparison_n_ensemble = table_comparison.loc[:,
                        table_comparison.columns[table_comparison.columns.str.endswith('ensemble')]]
wtl_df_n_ensemble_signif = pd.DataFrame()
methods = ['Standard_Bag_n_ensemble','Grouped_n_ensemble','Incremental_n_ensemble',
           'CDB_Host_n_ensemble',
           'CDB_kDN_n_ensemble','CDB_CLD_n_ensemble','CDB_LSC_n_ensemble',
           'CDB_N1_n_ensemble','CDB_N2_n_ensemble',
           'CDB_DCP_n_ensemble','CDB_TD_U_n_ensemble','CDB_F1_n_ensemble']
for method_i in methods[3:]:
    CM = method_i[len('CDB_'):-len('_n_ensemble')]
    wtl_df, _ = rank_df(table_comparison_n_ensemble, method_i, methods=methods)
    name_column = 'wtl_' + CM
    wtl_df_n_ensemble_signif[name_column] = "(" + wtl_df.apply(lambda row: ','.join(row.values.astype(str)), axis=1) + ")"

# Here we prefer to loss, it means the number of ensembles is lower
wtl_df_n_ensemble_signif.drop(['CDB_kDN_n_ensemble','CDB_CLD_n_ensemble','CDB_LSC_n_ensemble',
           'CDB_N1_n_ensemble','CDB_N2_n_ensemble',
           'CDB_DCP_n_ensemble','CDB_TD_U_n_ensemble','CDB_F1_n_ensemble'],inplace=True)




## Gráficos


# Seleccionar datos para mean
mean_cols = ['Standard_Bag_mean',
       'Grouped_Bag_mean',  'Incre_Bag_mean',
       'CDB_Host_mean',  'CDB_kDN_mean', 'CDB_CLD_mean',
             'CDB_LSC_mean', 'CDB_N1_mean',
       'CDB_N2_mean',  'CDB_DCP_mean','CDB_TD_U_mean',
             'CDB_F1_mean']
mean_data = table_comparison[mean_cols]


# Reestructurar los datos para cada métrica
mean_data = table_comparison[[col for col in table_comparison.columns if '_mean' in col]].melt(var_name='Method', value_name='Mean')
std_data = table_comparison[[col for col in table_comparison.columns if '_std' in col]].melt(var_name='Method', value_name='STD')
ensemble_data = table_comparison[[col for col in table_comparison.columns if '_n_ensemble' in col]].melt(var_name='Method', value_name='Num ensemble')


sns.boxplot(x='Method', y='Mean', data=mean_data,color='#D4D7FC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


sns.boxplot(x='Method', y='STD', data=std_data,color='#D4D7FC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

sns.boxplot(x='Method', y='Num ensemble', data=ensemble_data,color='#D4D7FC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

## En este caso, hacer plots no tiene mucho sentido porque los métodos de Mixed bagging van cambiando para cada caso
# pero bueno igual lo vamos a hacer


### Gráficos evolución



# Para CDB nos quedamos con el mejor valor y cogemos el resto de sus valores


list_datasets = best_param_cdb_signif.Dataset.unique()
CM = 'Hostility'
df_mixed.rename(columns={'n_trees':'n_ensemble'}, inplace=True)



for dataset_i in list_datasets:
    print(dataset_i)
    values = best_param_cdb_signif.loc[(best_param_cdb_signif['Dataset']==dataset_i) &
    (best_param_cdb_signif['weights']==CM),['alpha','split']]
    alpha = values.iloc[0]['alpha']
    split = values.iloc[0]['split']

    df_cdb_signif_plot = df_cdb_signif.loc[(df_cdb_signif['Dataset']==dataset_i) &
    (df_cdb_signif['weights']==CM) & (df_cdb_signif['alpha']==alpha) &
    (df_cdb_signif['split']==split),:]

    df_cdb_plot = df_cdb_f.loc[(df_cdb_f['Dataset']==dataset_i) &
    (df_cdb_f['weights']==CM) & (df_cdb_f['alpha']==alpha) &
    (df_cdb_f['split']==split),:]

    df_cdb_signif_plot['model'] = 'CDB_'+CM
    df_cdb_plot['model'] = 'CCDB'
    df_standard_plot = df_standard.loc[(df_standard['Dataset']==dataset_i),:]
    df_standard_plot['model'] = 'StandardBagg'
    df_mixed_plot = df_mixed.loc[(df_mixed['Dataset'] == dataset_i), :]


    df_plot = pd.concat([df_cdb_plot, df_cdb_signif_plot, df_standard_plot, df_mixed_plot])
    df_plot = df_plot[['n_ensemble', 'accuracy_mean', 'model']]  # Solo mantener las columnas relevantes

    # colors = {
    #     'CCDB': '#003366',  # Azul oscuro
    #     'CDB_'+CM: '#66b3ff',  # Azul claro
    #     'Grouped_Mixed_Bagging': '#F5AA0A',
    #     'Incremental_Mixed_Bagging': '#E10A45',
    #      'StandardBagg': '#1F2E2D'
    # }


    #plt.figure(figsize=(10, 6))
    for method, group in df_plot.groupby('model'):
        plt.plot(group['n_ensemble'], group['accuracy_mean'], marker='o', label=method)

    # Personalización del gráfico
    plt.title('Comparison of methods - Bagging '+dataset_i, fontsize=14)
    plt.xlabel('n_ensemble', fontsize=12)
    plt.ylabel('accuracy_mean', fontsize=12)
    plt.legend(title='Method', fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()


df_cdb_signif.columns # n_ensemble

df_standard.columns # n_ensemble

df_mixed.columns # n_trees





list_datasets = best_param_cdb.Dataset.unique()
CM = 'Hostility'
df_mixed.rename(columns={'n_trees':'n_ensemble'}, inplace=True)



for dataset_i in list_datasets:
    print(dataset_i)
    values = best_param_cdb.loc[(best_param_cdb['Dataset']==dataset_i) &
    (best_param_cdb['weights']==CM),['alpha','split']]
    alpha = values.iloc[0]['alpha']
    split = values.iloc[0]['split']

    df_cdb_signif_plot = df_cdb_f.loc[(df_cdb_f['Dataset']==dataset_i) &
    (df_cdb_f['weights']==CM) & (df_cdb_f['alpha']==str(alpha)) &
    (df_cdb_f['split']==str(split)),:]


    df_cdb_plot = df_cdb_f.loc[(df_cdb_f['Dataset']==dataset_i) &
    (df_cdb_f['weights']==CM) & (df_cdb_f['alpha']==alpha) &
    (df_cdb_f['split']==split),:]

    df_cdb_signif_plot['model'] = 'CDB_'+CM
    df_cdb_plot['model'] = 'CCDB'
    df_standard_plot = df_standard.loc[(df_standard['Dataset']==dataset_i),:]
    df_standard_plot['model'] = 'StandardBagg'
    df_mixed_plot = df_mixed.loc[(df_mixed['Dataset'] == dataset_i), :]


    df_plot = pd.concat([df_cdb_plot, df_cdb_signif_plot, df_standard_plot, df_mixed_plot])
    df_plot = df_plot[['n_ensemble', 'accuracy_mean', 'model']]  # Solo mantener las columnas relevantes

    # colors = {
    #     'CCDB': '#003366',  # Azul oscuro
    #     'CDB_'+CM: '#66b3ff',  # Azul claro
    #     'Grouped_Mixed_Bagging': '#F5AA0A',
    #     'Incremental_Mixed_Bagging': '#E10A45',
    #      'StandardBagg': '#1F2E2D'
    # }


    #plt.figure(figsize=(10, 6))
    for method, group in df_plot.groupby('model'):
        plt.plot(group['n_ensemble'], group['accuracy_mean'], marker='o', label=method)

    # Personalización del gráfico
    plt.title('Comparison of methods - Bagging '+dataset_i, fontsize=14)
    plt.xlabel('n_ensemble', fontsize=12)
    plt.ylabel('accuracy_mean', fontsize=12)
    plt.legend(title='Method', fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()




