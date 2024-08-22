## In this script we read, aggregate/summarize the results of ComplexityDrivenBagging
## for the different parameters tested in order to determine the best configuration of parameters
## for our method
## PARA EL CASO EN EL QUE TENEMOS CICLOS

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


##########################################################################
########                 STACK ALL AGGREGATED CSVs                ########
##########################################################################

# Como ahora tb queremos estudiar el número de ciclos, no podemos empezar agregando
# Así, comenzamos poniendo todos los resultados en el mismo dataset y luego ya vamos agregando

path_csv = os.chdir(root_path+'/Results_general_algorithm_cycles')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('Aggregated'):
        total_name_list.append(filename)

# len(total_name_list) # 7520

# General df to save all the results
cols = ['weights','n_cycle','n_ensemble', 'accuracy_mean', 'accuracy_std','Dataset','alpha','split']
df_total = pd.DataFrame(columns=cols)
i = 0
for data_file in total_name_list:
    i = i + 1
    # print(data_file)
    print(i)
    file = data_file
    name_data = data_file[data_file.find('AggregatedResults_CDB_') + len('AggregatedResults_CDB_'):data_file.rfind('_split')]
    alpha = data_file[data_file.find('alpha') + len('alpha'):data_file.rfind('.csv')]
    split = data_file[data_file.find('split') + len('split'):data_file.rfind('_alpha')]
    data = pd.read_csv(file)
    df_select = data[['weights','n_cycle','n_ensemble', 'accuracy_mean', 'accuracy_std']].copy()

    df_select['Dataset'] = name_data
    df_select['alpha'] = alpha
    df_select['split'] = split

    df_total = pd.concat([df_total,df_select])

# df_total.shape # 1404360

# Reorder columns
df_total = df_total.reindex(columns=['Dataset','weights', 'alpha', 'split','n_cycle','n_ensemble', 'accuracy_mean', 'accuracy_std'])
# To save the results
path_to_save = root_path+'/Results_general_algorithm_cycles'
os.chdir(path_to_save)
nombre_csv_agg = 'TotalAggregatedResults_ParameterConfiguration_CDB.csv'
# df_total.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)


# no podemos abrirlo completo en csv porque son demasiadas filas, pero si lo cargo, va bien
path_csv = os.chdir(root_path+'/Results_general_algorithm_cycles')
df_total = pd.read_csv('TotalAggregatedResults_ParameterConfiguration_CDB.csv')




#####################################################################################
########          GIVEN SPLIT S AND ALPHA, BEST NUMBER OF CYCLES?         ###########
#####################################################################################

# Para cada medida de complejidad
df_summary_CM = df_total.groupby(by=['weights','n_cycle','n_ensemble','alpha','split'], as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
df_summary_CM.columns = ['weights','n_cycle','n_ensemble','alpha','split','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
# Juntando todas las medidas de complejidad
df_summary = df_total.groupby(by=['n_cycle','n_ensemble','alpha','split'], as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
df_summary.columns = ['n_cycle','n_ensemble','alpha','split','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
# Máximo accuracy para cada combo split-alpha (para saber en qué ciclo se consigue)
aa = df_summary.groupby(by=['alpha','split'], as_index=False).idxmax('accuracy_mean_mean')

# Aquí vemos que en general sale el máximo por 300, es decir, el máximo número de modelos probados
df_max_acc = df_summary.loc[df_summary.reset_index().groupby(['alpha','split'])['accuracy_mean_mean'].idxmax()]
# Así, para cada como split-alpha vamos a buscar cuándo el accuracy deja de ser significativamente mayor




















##########################################################################
########              SUMMARIZE OF ALL AGGREGATED CSVs            ########
##########################################################################
# Quitamos n_ensemble = 0 y 9, para empezar en 20
# hacemos otro empezando en 30
out_values1 = [0,9]
out_values2 = [0,9,19]

path_csv = os.chdir(root_path+'/Results_general_algorithm/Aggregated_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('Aggregated'):
        total_name_list.append(filename)

# len(total_name_list) # 7520

# General df to save all the results
cols = ['weights','accuracy_mean_mean','accuracy_mean_median', 'accuracy_mean_std','Dataset','alpha','split']
df_total = pd.DataFrame(columns=cols)
i = 0
for data_file in total_name_list:
    i = i + 1
    # print(data_file)
    print(i)
    file = data_file
    name_data = data_file[data_file.find('AggregatedResults_CDB_') + len('AggregatedResults_CDB_'):data_file.rfind('_split')]
    alpha = data_file[data_file.find('alpha') + len('alpha'):data_file.rfind('.csv')]
    split = data_file[data_file.find('split') + len('split'):data_file.rfind('_alpha')]
    data = pd.read_csv(file)
    # data.shape
    # Quitamos los n_ensemble que no queremos
    # data = data[~data['n_ensemble'].isin(out_values1)] # from 20 ensembles
    data = data[~data['n_ensemble'].isin(out_values2)]  # from 30 ensembles
    df_summary = data.groupby(by='weights', as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
    df_summary.columns = ['weights','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
    df_summary['Dataset'] = name_data
    df_summary['alpha'] = alpha
    df_summary['split'] = split

    df_total = pd.concat([df_total,df_summary])

# df_total.shape # 67680

# Reorder columns
df_total = df_total.reindex(columns=['Dataset', 'alpha', 'split','weights', 'accuracy_mean_mean', 'accuracy_mean_median',
       'accuracy_mean_std'])
# To save the results
path_to_save = root_path+'/Results_general_algorithm'
os.chdir(path_to_save)
# nombre_csv_agg = 'SummarizeResults_ParameterConfiguration_CDB_from20ensembles.csv' # out_values1
nombre_csv_agg = 'SummarizeResults_ParameterConfiguration_CDB_from30ensembles.csv' # out_values2
df_total.to_csv(nombre_csv_agg, encoding='utf_8_sig', index=False)



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



#####################################################################################################################
###########                            ANALYSIS ACCORDING TO level of COMPLEXITY                          ###########
#####################################################################################################################
# df_total
df_complexity= pd.read_csv('complex_info_total.csv')
df_host = df_complexity[['Dataset','Hostility']].copy()
bins = [0, 0.15, 0.3, 2]
group_names = ['easy','medium','hard']
df_host['complexity'] = pd.cut(df_host['Hostility'], bins, labels=group_names, right=False)

df_total_complex = pd.merge(df_total, df_host, on='Dataset', how='outer')


#######################################################################################
#############                          LINEAR MODEL                       #############
#######################################################################################


### Mean ###
model = ols('accuracy_mean_mean ~ C(alpha) + C(split) + C(weights) + C(complexity)', data=df_total_complex)
fitted_model = model.fit()
fitted_model.summary()

anova_result = sm.stats.anova_lm(fitted_model, typ=2)
print(anova_result)

tukey = pairwise_tukeyhsd(endog=df_total_complex['accuracy_mean_mean'],
                          groups=df_total_complex['complexity'],
                          alpha=0.05)
print(tukey)



### Median ###
model = ols('accuracy_mean_median ~ C(alpha) + C(split) + C(weights) + C(complexity)', data=df_total_complex)
fitted_model = model.fit()
fitted_model.summary()

anova_result = sm.stats.anova_lm(fitted_model, typ=2)
print(anova_result)

tukey = pairwise_tukeyhsd(endog=df_total_complex['accuracy_mean_median'],
                          groups=df_total_complex['complexity'],
                          alpha=0.05)
print(tukey)



### STD ###
model = ols('accuracy_mean_std ~ C(alpha) + C(split) + C(weights) + C(complexity)', data=df_total_complex)
fitted_model = model.fit()
fitted_model.summary()

anova_result = sm.stats.anova_lm(fitted_model, typ=2)
print(anova_result)


import pingouin as pg
res = pg.rm_anova(dv='accuracy_mean_mean', within=['weights','alpha','split'], subject='Dataset',
                  data=df_total_complex, detailed=True)
res

# Mauchly’s test of sphericity
pg.sphericity(data=df_total_complex, dv='accuracy_mean_mean', subject='Dataset', within='alpha')

### Alpha
df_total_complex.groupby('alpha').var()


# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_total_complex['accuracy_mean_std'],
                          groups=df_total_complex['alpha'],
                          alpha=0.05)
print(tukey)

# pw = fitted_model.t_test_pairwise("C(alpha)")
# aa = pw.result_frame
#
#
# df_total_complex.groupby('alpha').mean()

tukey = pairwise_tukeyhsd(endog=df_total_complex['accuracy_mean_std'],
                          groups=df_total_complex['split'],
                          alpha=0.05)
print(tukey)


tukey = pairwise_tukeyhsd(endog=df_total_complex['accuracy_mean_std'],
                          groups=df_total_complex['weights'],
                          alpha=0.05)
print(tukey)

tukey = pairwise_tukeyhsd(endog=df_total_complex['accuracy_mean_std'],
                          groups=df_total_complex['complexity'],
                          alpha=0.05)
print(tukey)


### Heatmap per complexity measure and per type of dataset according to its complexity

CM = 'Hostility'
df_total_hard = df_total_complex.loc[(df_total_complex['weights'] == CM) & (df_total_complex['complexity'] == 'hard'), :]
df_total_medium = df_total_complex.loc[(df_total_complex['weights'] == CM) & (df_total_complex['complexity'] == 'medium'), :]
df_total_easy = df_total_complex.loc[(df_total_complex['weights'] == CM) & (df_total_complex['complexity'] == 'easy'), :]



summary_hard = df_total_hard.groupby(['alpha', 'split'], as_index=False)[['accuracy_mean_mean',
                                                                          'accuracy_mean_median',
                                                                          'accuracy_mean_std']].mean()

summary_medium = df_total_medium.groupby(['alpha', 'split'], as_index=False)[['accuracy_mean_mean',
                                                                          'accuracy_mean_median',
                                                                          'accuracy_mean_std']].mean()

summary_easy = df_total_easy.groupby(['alpha', 'split'], as_index=False)[['accuracy_mean_mean',
                                                                          'accuracy_mean_median',
                                                                          'accuracy_mean_std']].mean()

df_to_plot_mean_h = summary_hard.pivot(index='alpha', columns='split', values='accuracy_mean_mean')
df_to_plot_mean_h.sort_index(level=0, inplace=True, ascending=False)
df_to_plot_mean_m = summary_medium.pivot(index='alpha', columns='split', values='accuracy_mean_mean')
df_to_plot_mean_m.sort_index(level=0, inplace=True, ascending=False)
df_to_plot_mean_e = summary_easy.pivot(index='alpha', columns='split', values='accuracy_mean_mean')
df_to_plot_mean_e.sort_index(level=0, inplace=True, ascending=False)

df_to_plot_median_h = summary_hard.pivot(index='alpha', columns='split', values='accuracy_mean_median')
df_to_plot_median_h.sort_index(level=0, inplace=True, ascending=False)
df_to_plot_median_m = summary_medium.pivot(index='alpha', columns='split', values='accuracy_mean_median')
df_to_plot_median_m.sort_index(level=0, inplace=True, ascending=False)
df_to_plot_median_e = summary_easy.pivot(index='alpha', columns='split', values='accuracy_mean_median')
df_to_plot_median_e.sort_index(level=0, inplace=True, ascending=False)

df_to_plot_std_h = summary_hard.pivot(index='alpha', columns='split', values='accuracy_mean_std')
df_to_plot_std_h.sort_index(level=0, inplace=True, ascending=False)
df_to_plot_std_m = summary_medium.pivot(index='alpha', columns='split', values='accuracy_mean_std')
df_to_plot_std_m.sort_index(level=0, inplace=True, ascending=False)
df_to_plot_std_e = summary_easy.pivot(index='alpha', columns='split', values='accuracy_mean_std')
df_to_plot_std_e.sort_index(level=0, inplace=True, ascending=False)

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
sns.heatmap(df_to_plot_mean_h, cmap="YlGnBu",ax=axes[0,0]).set(title='Hard Mean')
sns.heatmap( df_to_plot_median_h, cmap="YlGnBu",ax=axes[0,1]).set(title='Hard Median')
sns.heatmap( df_to_plot_std_h, cmap="YlGnBu", ax=axes[0,2]).set(title='Hard STD')

sns.heatmap(df_to_plot_mean_m, cmap="YlGnBu", ax=axes[1,0]).set(title='Medium Mean')
sns.heatmap( df_to_plot_median_m, cmap="YlGnBu", ax=axes[1,1]).set(title='Medium Median')
sns.heatmap( df_to_plot_std_m, cmap="YlGnBu", ax=axes[1,2]).set(title='Medium STD')

sns.heatmap(df_to_plot_mean_e, cmap="YlGnBu", ax=axes[2,0]).set(title='Easy Mean')
sns.heatmap( df_to_plot_median_e, cmap="YlGnBu", ax=axes[2,1]).set(title='Easy Median')
sns.heatmap( df_to_plot_std_e, cmap="YlGnBu", ax=axes[2,2]).set(title='Easy STD')
plt.show()



#####################################################################################################################
###########                                   ANALYSIS WITH WIN-TIE-LOSS                                  ###########
#####################################################################################################################

path_csv = os.chdir(root_path+'/Results_general_algorithm')
df_total = pd.read_csv('SummarizeResults_ParameterConfiguration_CDB.csv')

# df_total
df_complexity= pd.read_csv('complex_info_total.csv')
df_host = df_complexity[['Dataset','Hostility']].copy()
bins = [0, 0.15, 0.3, 2]
group_names = ['easy','medium','hard']
df_host['complexity'] = pd.cut(df_host['Hostility'], bins, labels=group_names, right=False)

df_total_complex = pd.merge(df_total, df_host, on='Dataset', how='outer')

df_rank = df_total_complex.copy()

list_datasets = list(np.unique(df_total['Dataset']))
list_CM = list(np.unique(df_total['weights']))

for dataset in list_datasets:
    for CM in list_CM:
        filter = (df_rank['Dataset'] == dataset) & (df_rank['weights'] == CM)

        df_rank.loc[filter,'rank_mean_accuracy'] = df_rank.loc[filter,'accuracy_mean_mean'].rank(ascending=False,method='min')
        df_rank.loc[filter,'rank_median_accuracy'] = df_rank.loc[filter,'accuracy_mean_median'].rank(ascending=False,method='min')
        df_rank.loc[filter,'rank_std_accuracy'] = df_rank.loc[filter,'accuracy_mean_std'].rank(ascending=True,method='min')



