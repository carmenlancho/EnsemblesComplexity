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

# Para cada medida de complejidad y dataset
df_summary_CM = df_total.groupby(by=['weights','n_cycle','n_ensemble','alpha','split'], as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
df_summary_CM.columns = ['weights','n_cycle','n_ensemble','alpha','split','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
# Agregamos medidas de complejidad y no datasets
df_summary_data = df_total.groupby(by=['Dataset','n_cycle','n_ensemble','alpha','split'], as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
df_summary_data.columns = ['Dataset','n_cycle','n_ensemble','alpha','split','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
# Comienzo juntando alpha-split
df_summary_data["combo_alpha_split"] = 'alpha'+df_summary_data["alpha"].astype(str) + '-split' + df_summary_data["split"].astype(str)

# Juntando todas las medidas de complejidad
df_summary = df_total.groupby(by=['n_cycle','n_ensemble','alpha','split'], as_index=False).agg({'accuracy_mean': [np.mean, np.median, np.std]})
df_summary.columns = ['n_cycle','n_ensemble','alpha','split','accuracy_mean_mean','accuracy_mean_median',  'accuracy_mean_std']
# Máximo accuracy para cada combo split-alpha (para saber en qué ciclo se consigue)
# Aquí vemos que en general sale el máximo por 300, es decir, el máximo número de modelos probados
df_max_acc = df_summary.loc[df_summary.reset_index().groupby(['alpha','split'])['accuracy_mean_mean'].idxmax()]
# Así, para cada como split-alpha vamos a buscar cuándo el accuracy deja de ser significativamente mayor


# Comienzo juntando alpha-split
df_summary["combo_alpha_split"] = 'alpha'+df_summary["alpha"].astype(str) + '-split' + df_summary["split"].astype(str)


### Finalmente realizamos el análisis de parámetros en R y las conclusiones son

# En el archivo CDB_cycles_ParametersComboAlphaSplit_dif_no_signif_cycles_mean_median_std_num_models. csv
# tenemos el máximo número de modelos y de ciclos en función de las comparaciones múltiples del num de ciclos
# split Domain: [2, 4, 6, 8, 10, 12, 14]
# alpha Domain: [2, 4, 6, 8, 10]













