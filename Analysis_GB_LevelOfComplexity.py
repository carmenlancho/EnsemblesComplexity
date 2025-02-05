########################################################################################
######### Script para analizar los resultados del gradient boosting por nivel de complejidad
######## Simplemente es porque es más cómodo trabajar aquí que directamente en el Notebook
####### pero los resultados finales los pondremos allí para analizarlos con mayor comodidad


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import math


root_path = os.getcwd()

path_csv = os.path.join(root_path, 'Results_GB')
os.chdir(path_csv)


colour_palette_personalized = {
    "classic": "#FFD700",   # yellow
    "sample_weight_easy": "#C7F7FF", # blue
    "sample_weight_easy_x2": "#00CED1", # blue
    "sample_weight_hard": "#FFB3DA",    # magenta
    "sample_weight_hard_x2": "#FF1493",    # magenta
    "Classic": "#FFD700",   # yellow
    "Easy": "#C7F7FF", # blue
     "Easy x2": "#00CED1", # blue
    "Hard": "#FFB3DA",    # magenta
     "Hard x2": "#FF1493"    # magenta
}



specific_path = os.path.join(path_csv, '*Aggregated*.csv')
selected_files = glob.glob(specific_path)
all_datasets = pd.concat([pd.read_csv(f) for f in selected_files], ignore_index=True)


path_complex = os.path.join(root_path, 'datasets/complexity_info')
os.chdir(path_complex)
df_complex = pd.read_csv('complex_info_dataset_20250115.csv')
df_complex.head()



def win_tie_loss_comparison_info_complexity(data, complexity_df, main_method, compare_method, loss_function, metric='test_acc_mean',
                                            n_ensemble_values=[10, 25, 50, 100, 150, 200, 250, 300]):
    """
    Realiza un análisis win-tie-loss comparando el métod principal con otro métod específico para cada medida de complejidad.

    Parameters:
    - data: DataFrame con las columnas ['dataset', 'n_ensemble', 'method_weights', 'compl_measure', metric]
    - main_method: métod principal que se desea comparar (ejemplo: 'classic')
    - compare_method: métod específico con el cual comparar el principal (ejemplo: 'init_easy')
    - metric: métrica de comparación (por defecto 'test_acc_mean')
    - n_ensemble_values: valores de n_ensemble a considerar

    Returns:
    - Una tabla con el conteo de wins, ties, y losses por cada valor de n_ensemble y medida de complejidad.
    """
    results = []
    # Crear una lista para almacenar los resultados
    all_results = []
    value_wtl = 'nada'

    for n in n_ensemble_values:
        # Filtrar los datos para el valor actual de n_ensemble
        subset_n = data[(data['n_ensemble'] == n) & (data['loss_selected'] == loss_function)]


        # Crear un diccionario para almacenar los resultados de esta combinación de n_ensemble
        row = {'n_ensemble': n}

        CM_list = subset_n['compl_measure'].unique()[1:] # to delete none

        for compl in CM_list:
            win, tie, loss = 0, 0, 0

            # Filtrar los datos para la medida de complejidad actual
            subset_compl = subset_n[subset_n['compl_measure'] == compl]
            #if (main_method == 'classic') or (compare_method == 'classic'):
            #     subset_compl_main = subset_n[subset_n['compl_measure'] == 'none']

            for dataset in subset_compl['dataset'].unique():
                # Filtrar para el dataset y métod en cuestión

                if (main_method == 'classic'):
                    subset_compl_main = subset_n[subset_n['compl_measure'] == 'none']
                    main_value = subset_compl_main[(subset_compl_main['dataset'] == dataset) & (subset_compl_main['method_weights'] == main_method)][metric].values
                else:
                    main_value = subset_compl[(subset_compl['dataset'] == dataset) & (subset_compl['method_weights'] == main_method)][metric].values
                if (compare_method == 'classic'):
                    subset_compl_compare = subset_n[subset_n['compl_measure'] == 'none']
                    compare_value = subset_compl_compare[(subset_compl_compare['dataset'] == dataset) & (subset_compl_compare['method_weights'] == compare_method)][metric].values
                else:
                    compare_value = subset_compl[(subset_compl['dataset'] == dataset) & (subset_compl['method_weights'] == compare_method)][metric].values

                # Verificar que tenemos valores únicos para cada dataset y métod
                if main_value.size > 0 and compare_value.size > 0:
                    main_value = main_value[0]
                    compare_value = compare_value[0]

                    # Comparación win-tie-loss
                    if main_value < compare_value:
                        win += 1
                        value_wtl = 'win'
                    elif main_value == compare_value:
                        tie += 1
                        value_wtl = 'tie'
                    else:
                        loss += 1
                        value_wtl = 'loss'

                # Obtener las medidas de complejidad para el dataset
                complexity_values = complexity_df[complexity_df['dataset'] == dataset][compl].values
                if complexity_values.size > 0:
                    complexity_value = complexity_values[0]

                    # Almacenar los resultados en una lista de diccionarios
                    all_results.append({
                        'n_ensemble': n,
                        'compl_measure': compl,
                        'result': value_wtl,
                        'complexity_value': complexity_value,
                        'dataset_name':dataset,
                        main_method:main_value,
                        compare_method:compare_value
                    })

            # Guardar el resultado para esta medida de complejidad en una columna específica
            row[f'{compare_method}_{compl}'] = (win, tie, loss)

        # Agregar el resultado de esta iteración a los resultados
        results.append(row)

    # Convertir los resultados a DataFrame
    win_tie_loss_df = pd.DataFrame(results)

    return win_tie_loss_df, all_results



# Función para categorizar en tres niveles
def categorize_column(column):
    min_val = column.min()
    max_val = column.max()
    thresholds = np.linspace(min_val, max_val, 4)  # Dividimos el rango en 3 categorías
    return pd.cut(
        column,
        bins=thresholds,
        labels=["low", "medium", "high"],
        include_lowest=True
    )

def summary_CM_complexity_dataset(CM,dataset,df_complexity, loss_function): #dataset es all_datasets
    # nos quedamos con las filas referentes a la CM en particular, al modo clásico
    # y con la función de pérdida escogida
    all_datasets_CM = dataset.loc[((dataset.compl_measure == CM) | (dataset.compl_measure == 'none')) & (dataset.loss_selected == loss_function),:]

    df_complexity2 = df_complexity.copy()

    # Complejidad de los datasets
    for col in df_complexity2.columns[1:]:  # Excluye la columna 'dataset'
        df_complexity2[f'cat_{col}'] = categorize_column(df_complexity2[col])

    # We introduce the complexity of all datasets in the main df
    df_merged = all_datasets_CM.merge(df_complexity2[['dataset', 'cat_'+CM]], on='dataset', how='left')

    # Filtrar en función de 'cat_CMi'
    df_low_CM = df_merged[df_merged['cat_'+CM] == 'low']
    df_medium_CM = df_merged[df_merged['cat_'+CM] == 'medium']
    df_high_CM = df_merged[df_merged['cat_'+CM] == 'high']

    summary_results_low = df_low_CM.groupby(['dataset', 'method_weights']).agg(
        test_acc_mean_median=('test_acc_mean', 'median'),
        test_acc_mean_std=('test_acc_mean', 'std'),
        test_acc_mean_total_mean=('test_acc_mean', 'mean')
    ).reset_index()

    summary_results_medium = df_medium_CM.groupby(['dataset', 'method_weights']).agg(
        test_acc_mean_median=('test_acc_mean', 'median'),
        test_acc_mean_std=('test_acc_mean', 'std'),
        test_acc_mean_total_mean=('test_acc_mean', 'mean')
    ).reset_index()

    summary_results_high = df_high_CM.groupby(['dataset', 'method_weights']).agg(
        test_acc_mean_median=('test_acc_mean', 'median'),
        test_acc_mean_std=('test_acc_mean', 'std'),
        test_acc_mean_total_mean=('test_acc_mean', 'mean')
    ).reset_index()

    # Pivotar para obtener una tabla más organizada
    summary_pivot_low = summary_results_low.pivot(index='dataset', columns='method_weights',
                                          values=['test_acc_mean_median', 'test_acc_mean_std', 'test_acc_mean_total_mean'])

    summary_pivot_medium = summary_results_medium.pivot(index='dataset', columns='method_weights',
                                          values=['test_acc_mean_median', 'test_acc_mean_std', 'test_acc_mean_total_mean'])

    summary_pivot_high = summary_results_high.pivot(index='dataset', columns='method_weights',
                                          values=['test_acc_mean_median', 'test_acc_mean_std', 'test_acc_mean_total_mean'])

    # Renombrar columnas para que sean más fáciles de leer
    summary_pivot_low.columns = [f'{method}_{stat}' for stat, method in summary_pivot_low.columns]
    summary_pivot_low.reset_index(inplace=True)
    summary_pivot_medium.columns = [f'{method}_{stat}' for stat, method in summary_pivot_medium.columns]
    summary_pivot_medium.reset_index(inplace=True)
    summary_pivot_high.columns = [f'{method}_{stat}' for stat, method in summary_pivot_high.columns]
    summary_pivot_high.reset_index(inplace=True)


    ###----- WTL analysis -----###
    wtl_sw_easy_low, _ = win_tie_loss_comparison_info_complexity(df_low_CM, df_complexity, main_method='classic', compare_method='sample_weight_easy',
                                                                       loss_function=loss_function)
    wtl_sw_easy_medium, _ = win_tie_loss_comparison_info_complexity(df_medium_CM, df_complexity, main_method='classic', compare_method='sample_weight_easy', loss_function=loss_function)
    wtl_sw_easy_high, _ = win_tie_loss_comparison_info_complexity(df_high_CM, df_complexity, main_method='classic', compare_method='sample_weight_easy', loss_function=loss_function)

    wtl_sw_easy_x2_low, _ = win_tie_loss_comparison_info_complexity(df_low_CM, df_complexity, main_method='classic', compare_method='sample_weight_easy_x2',
                                                                       loss_function=loss_function)
    wtl_sw_easy_x2_medium, _ = win_tie_loss_comparison_info_complexity(df_medium_CM, df_complexity, main_method='classic', compare_method='sample_weight_easy_x2', loss_function=loss_function)
    wtl_sw_easy_x2_high, _ = win_tie_loss_comparison_info_complexity(df_high_CM, df_complexity, main_method='classic', compare_method='sample_weight_easy_x2', loss_function=loss_function)

    wtl_sw_hard_low, _ = win_tie_loss_comparison_info_complexity(df_low_CM, df_complexity, main_method='classic', compare_method='sample_weight_hard',
    loss_function=loss_function)
    wtl_sw_hard_medium, _ = win_tie_loss_comparison_info_complexity(df_medium_CM, df_complexity, main_method='classic', compare_method='sample_weight_hard', loss_function=loss_function)
    wtl_sw_hard_high, _ = win_tie_loss_comparison_info_complexity(df_high_CM, df_complexity, main_method='classic', compare_method='sample_weight_hard', loss_function=loss_function)

    wtl_sw_hard_x2_low, _ = win_tie_loss_comparison_info_complexity(df_low_CM, df_complexity, main_method='classic', compare_method='sample_weight_hard_x2',
    loss_function=loss_function)
    wtl_sw_hard_x2_medium, _ = win_tie_loss_comparison_info_complexity(df_medium_CM, df_complexity, main_method='classic', compare_method='sample_weight_hard_x2', loss_function=loss_function)
    wtl_sw_hard_x2_high, _ = win_tie_loss_comparison_info_complexity(df_high_CM, df_complexity, main_method='classic', compare_method='sample_weight_hard_x2', loss_function=loss_function)


    ###------  Plots  ------###
    # Low complexity dataset
    summary_long_low = summary_pivot_low.melt(id_vars='dataset',
                                         value_vars=[
                                            'classic_test_acc_mean_median',
                                            'sample_weight_easy_test_acc_mean_median',
                                            'sample_weight_hard_test_acc_mean_median',
                                             'sample_weight_easy_x2_test_acc_mean_median',
                                            'sample_weight_hard_x2_test_acc_mean_median',
                                            'classic_test_acc_mean_std',
                                            'sample_weight_easy_test_acc_mean_std',
                                            'sample_weight_hard_test_acc_mean_std',
                                            'sample_weight_easy_x2_test_acc_mean_std',
                                            'sample_weight_hard_x2_test_acc_mean_std',
                                            'classic_test_acc_mean_total_mean',
                                            'sample_weight_easy_test_acc_mean_total_mean',
                                            'sample_weight_hard_test_acc_mean_total_mean',
                                             'sample_weight_easy_x2_test_acc_mean_total_mean',
                                            'sample_weight_hard_x2_test_acc_mean_total_mean'
                                        ],
                                        var_name='method_stat',
                                        value_name='value')

    # Separar los nombres de métod y estadístico
    summary_long_low[['method', 'statistic']] = summary_long_low['method_stat'].str.rsplit('_', n=1, expand=True)
    summary_long_low['method'] = summary_long_low['method'].replace({
         'classic_test_acc_mean': 'Classic',
        'sample_weight_easy_test_acc_mean': 'Easy',
        'sample_weight_easy_x2_test_acc_mean': 'Easy x2',
        'sample_weight_hard_test_acc_mean': 'Hard',
        'sample_weight_hard_x2_test_acc_mean': 'Hard x2',
        'classic_test_acc_mean_total': 'Classic',
        'sample_weight_easy_test_acc_mean_total': 'Easy',
        'sample_weight_easy_x2_test_acc_mean_total': 'Easy x2',
        'sample_weight_hard_test_acc_mean_total': 'Hard',
        'sample_weight_hard_x2_test_acc_mean_total': 'Hard x2'
    })


    # Crear un boxplot para cada estadístico
    plt.figure(figsize=(12, 5))

    # Usar un bucle para crear un boxplot por cada estadístico
    for i, stat in enumerate(summary_long_low['statistic'].unique(), start=1):
        plt.subplot(1, 3, i)  # 3 filas, 1 columna
        sns.boxplot(data=summary_long_low[summary_long_low['statistic'] == stat],
                     x='method',
                     y='value',
                     palette=colour_palette_personalized)

        plt.xlabel('Method')
        plt.ylabel(stat.capitalize())

    # Título global para el gráfico
    plt.suptitle('Boxplots of Summary Statistics Low complexity Datasets '+CM, fontsize=16)


    plt.tight_layout()
    plt.show()

    print('Low complexity datasets: WTL Classic vs sw_easy')
    print(wtl_sw_easy_low)
    print('Low complexity datasets: WTL Classic vs sw_easy_x2')
    print(wtl_sw_easy_x2_low)
    print('Low complexity datasets: WTL Classic vs sw_hard')
    print(wtl_sw_hard_low)
    print('Low complexity datasets: WTL Classic vs sw_hard_x2')
    print(wtl_sw_hard_x2_low)


    # Medium complexity dataset
    summary_long_medium = summary_pivot_medium.melt(id_vars='dataset',
                                         value_vars=[
                                            'classic_test_acc_mean_median',
                                            'sample_weight_easy_test_acc_mean_median',
                                            'sample_weight_hard_test_acc_mean_median',
                                             'sample_weight_easy_x2_test_acc_mean_median',
                                            'sample_weight_hard_x2_test_acc_mean_median',
                                            'classic_test_acc_mean_std',
                                            'sample_weight_easy_test_acc_mean_std',
                                            'sample_weight_hard_test_acc_mean_std',
                                            'sample_weight_easy_x2_test_acc_mean_std',
                                            'sample_weight_hard_x2_test_acc_mean_std',
                                            'classic_test_acc_mean_total_mean',
                                            'sample_weight_easy_test_acc_mean_total_mean',
                                            'sample_weight_hard_test_acc_mean_total_mean',
                                             'sample_weight_easy_x2_test_acc_mean_total_mean',
                                            'sample_weight_hard_x2_test_acc_mean_total_mean'
                                        ],
                                        var_name='method_stat',
                                        value_name='value')

    # Separar los nombres de métod y estadístico
    summary_long_medium[['method', 'statistic']] = summary_long_medium['method_stat'].str.rsplit('_', n=1, expand=True)
    summary_long_medium['method'] = summary_long_medium['method'].replace({
         'classic_test_acc_mean': 'Classic',
        'sample_weight_easy_test_acc_mean': 'Easy',
        'sample_weight_easy_x2_test_acc_mean': 'Easy x2',
        'sample_weight_hard_test_acc_mean': 'Hard',
        'sample_weight_hard_x2_test_acc_mean': 'Hard x2',
        'classic_test_acc_mean_total': 'Classic',
        'sample_weight_easy_test_acc_mean_total': 'Easy',
        'sample_weight_easy_x2_test_acc_mean_total': 'Easy x2',
        'sample_weight_hard_test_acc_mean_total': 'Hard',
        'sample_weight_hard_x2_test_acc_mean_total': 'Hard x2'
    })


    # Crear un boxplot para cada estadístico
    plt.figure(figsize=(12, 5))

    # Usar un bucle para crear un boxplot por cada estadístico
    for i, stat in enumerate(summary_long_medium['statistic'].unique(), start=1):
        plt.subplot(1, 3, i)  # 3 filas, 1 columna
        sns.boxplot(data=summary_long_medium[summary_long_medium['statistic'] == stat],
                     x='method',
                     y='value',
                     palette=colour_palette_personalized)

        plt.xlabel('Method')
        plt.ylabel(stat.capitalize())

    # Título global para el gráfico
    plt.suptitle('Boxplots of Summary Statistics Medium complexity Datasets '+CM, fontsize=16)


    plt.tight_layout()
    plt.show()

    print('Medium complexity datasets: WTL Classic vs sw_easy')
    print(wtl_sw_easy_medium)
    print('Medium complexity datasets: WTL Classic vs sw_easy_x2')
    print(wtl_sw_easy_x2_medium)
    print('Medium complexity datasets: WTL Classic vs sw_hard')
    print(wtl_sw_hard_medium)
    print('Medium complexity datasets: WTL Classic vs sw_hard_x2')
    print(wtl_sw_hard_x2_medium)



    # High complexity
    summary_long_high = summary_pivot_high.melt(id_vars='dataset',
                                         value_vars=[
                                            'classic_test_acc_mean_median',
                                            'sample_weight_easy_test_acc_mean_median',
                                            'sample_weight_hard_test_acc_mean_median',
                                             'sample_weight_easy_x2_test_acc_mean_median',
                                            'sample_weight_hard_x2_test_acc_mean_median',
                                            'classic_test_acc_mean_std',
                                            'sample_weight_easy_test_acc_mean_std',
                                            'sample_weight_hard_test_acc_mean_std',
                                            'sample_weight_easy_x2_test_acc_mean_std',
                                            'sample_weight_hard_x2_test_acc_mean_std',
                                            'classic_test_acc_mean_total_mean',
                                            'sample_weight_easy_test_acc_mean_total_mean',
                                            'sample_weight_hard_test_acc_mean_total_mean',
                                             'sample_weight_easy_x2_test_acc_mean_total_mean',
                                            'sample_weight_hard_x2_test_acc_mean_total_mean'
                                        ],
                                        var_name='method_stat',
                                        value_name='value')

    # Separar los nombres de métod y estadístico
    summary_long_high[['method', 'statistic']] = summary_long_high['method_stat'].str.rsplit('_', n=1, expand=True)
    summary_long_high['method'] = summary_long_high['method'].replace({
        'classic_test_acc_mean': 'Classic',
        'sample_weight_easy_test_acc_mean': 'Easy',
        'sample_weight_easy_x2_test_acc_mean': 'Easy x2',
        'sample_weight_hard_test_acc_mean': 'Hard',
        'sample_weight_hard_x2_test_acc_mean': 'Hard x2',
        'classic_test_acc_mean_total': 'Classic',
        'sample_weight_easy_test_acc_mean_total': 'Easy',
        'sample_weight_easy_x2_test_acc_mean_total': 'Easy x2',
        'sample_weight_hard_test_acc_mean_total': 'Hard',
        'sample_weight_hard_x2_test_acc_mean_total': 'Hard x2'
    })


    # Crear un boxplot para cada estadístico
    plt.figure(figsize=(12, 5))

    # Usar un bucle para crear un boxplot por cada estadístico
    for i, stat in enumerate(summary_long_high['statistic'].unique(), start=1):
        plt.subplot(1, 3, i)  # 3 filas, 1 columna
        sns.boxplot(data=summary_long_high[summary_long_high['statistic'] == stat],
                     x='method',
                     y='value',
                     palette=colour_palette_personalized)

        plt.xlabel('Method')
        plt.ylabel(stat.capitalize())

    # Título global para el gráfico
    plt.suptitle('Boxplots of Summary Statistics High complexity Datasets '+CM, fontsize=16)


    plt.tight_layout()
    plt.show()

    print('High complexity datasets: WTL Classic vs sw_easy')
    print(wtl_sw_easy_high)
    print('High complexity datasets: WTL Classic vs sw_easy_x2')
    print(wtl_sw_easy_x2_high)
    print('High complexity datasets: WTL Classic vs sw_hard')
    print(wtl_sw_hard_high)
    print('High complexity datasets: WTL Classic vs sw_x2_hard')
    print(wtl_sw_hard_x2_high)



    return summary_pivot_low, summary_pivot_medium, summary_pivot_high, wtl_sw_easy_low, wtl_sw_easy_medium, wtl_sw_easy_high, wtl_sw_easy_x2_low, wtl_sw_easy_x2_medium,wtl_sw_easy_x2_high, wtl_sw_hard_low, wtl_sw_hard_medium, wtl_sw_hard_high, wtl_sw_hard_x2_low, wtl_sw_hard_x2_medium, wtl_sw_hard_x2_high

win_tie_loss_df, all_results = win_tie_loss_comparison_info_complexity(all_datasets, df_complex, loss_function='exponential', main_method='classic', compare_method='sample_weight_easy')

results_df = pd.DataFrame(all_results)
results_df


# Categorizar la columna 'classic'
bins = [0, 0.75, 0.85, 1]
labels = ['Acc < 0.75', 'Acc 0.75-0.85', 'Acc > 0.85']
results_df['classic_category'] = pd.cut(results_df['classic'], bins=bins, labels=labels)
results_df


# Calcular la distribución de 'result' por categoría, n_ensemble y compl_measure
distribution = (
    results_df
    .groupby(['classic_category', 'n_ensemble', 'compl_measure', 'result'])
    .size()
    .reset_index(name='count')  # Contar las ocurrencias
)
distribution


# Calcular proporciones dentro de cada grupo
distribution['proportion'] = (
    distribution['count'] /
    distribution.groupby(['classic_category', 'n_ensemble', 'compl_measure'])['count'].transform('sum')
)

# Mostrar la tabla de resultados
print(distribution)


df_for_max = all_datasets.loc[(all_datasets['loss_selected'] == 'exponential') & ((all_datasets['method_weights'] == 'classic') | (all_datasets['method_weights'] == 'sample_weight_easy')),
['dataset','n_ensemble','method_weights','compl_measure','test_acc_mean','test_acc_std']]
df_for_max


# Agrupar por dataset, method_weights y compl_measure, y encontrar el máximo de test_acc_mean y el n_ensemble correspondiente
max_performance = (
    df_for_max
    .groupby(['dataset', 'method_weights', 'compl_measure'], as_index=False)
    .apply(lambda group: group.loc[group['test_acc_mean'].idxmax(), ['dataset', 'method_weights', 'compl_measure', 'n_ensemble', 'test_acc_mean']])
    .reset_index(drop=True)
)

# Renombrar las columnas
max_performance.rename(columns={
    'n_ensemble': 'n_ensemble_max',
    'test_acc_mean': 'test_acc_mean_max'
}, inplace=True)

max_performance


# Categorizar la dificultad (aquí estamos categorizando en función del máximo accuracy que logra classic GB)
max_performance['classic_category'] = pd.cut(
    max_performance.loc[max_performance['method_weights'] == 'classic', 'test_acc_mean_max'],
    bins = [0, 0.75, 0.85, 1],
    labels = ['Acc < 0.75', 'Acc 0.75-0.85', 'Acc > 0.85'],
    include_lowest=True
)

# Crear un dataframe intermedio con las categorías solo para el métod clásico
classic_categories = max_performance[max_performance['method_weights'] == 'classic'][['dataset', 'compl_measure', 'classic_category']]

# Asegurarnos de no tener duplicados en las categorías
classic_categories = classic_categories.drop_duplicates(subset=['dataset'])

# Merge para asignar la categoría a todos los métodos
max_performance = max_performance.merge(
    classic_categories[['dataset', 'classic_category']],
    on=['dataset'],
    how='left'
)
max_performance.drop('classic_category_x', axis=1, inplace=True)

# lo vemos numéricamente
boxplot_stats = (
    max_performance
    .groupby(["classic_category_y", "compl_measure"])["n_ensemble_max"]
    .describe()[["min", "25%", "50%", "75%", "max"]]
    .rename(columns={"25%": "Q1", "50%": "median", "75%": "Q3"})
    .reset_index()
)

boxplot_stats


# Num de datasets por categoría
max_performance.head()
max_performance['classic_category_y'].value_counts()/10

# lo vemos numéricamente
boxplot_stats = (
    max_performance
    .groupby(["classic_category_y", "compl_measure"])["test_acc_mean_max"]
    .describe()[["min", "25%", "50%", "75%", "max"]]
    .rename(columns={"25%": "Q1", "50%": "median", "75%": "Q3"})
    .reset_index()
)

boxplot_stats


classic_categories.drop(['compl_measure'], axis=1, inplace=True)
# Añadimos la ctegorización al dataset filtrado (para tener solo exponential loss, sample_weitght_easy y classic)
df_hardGB = df_for_max.merge(
    classic_categories[['dataset', 'classic_category']],
    on=['dataset'],
    how='left'
)



df_hardGB2 = df_hardGB.loc[df_hardGB['classic_category']=='Acc < 0.75',:]
df_hardGB2



def win_tie_loss_comparison_info_complexity_hardGB(data, complexity_df, main_method, compare_method, metric='test_acc_mean',
                                            n_ensemble_values=[10, 25, 50, 100, 150, 200, 250, 300]):
    """
    Realiza un análisis win-tie-loss comparando el método principal con otro método específico para cada medida de complejidad.

    Parameters:
    - data: DataFrame con las columnas ['dataset', 'n_ensemble', 'method_weights', 'compl_measure', metric]
    - main_method: método principal que se desea comparar (ejemplo: 'classic')
    - compare_method: método específico con el cual comparar el principal (ejemplo: 'init_easy')
    - metric: métrica de comparación (por defecto 'test_acc_mean')
    - n_ensemble_values: valores de n_ensemble a considerar

    Returns:
    - Una tabla con el conteo de wins, ties, y losses por cada valor de n_ensemble y medida de complejidad.
    """
    results = []
    # Crear una lista para almacenar los resultados
    all_results = []
    value_wtl = 'nada'

    for n in n_ensemble_values:
        # Filtrar los datos para el valor actual de n_ensemble
        subset_n = data[(data['n_ensemble'] == n)]


        # Crear un diccionario para almacenar los resultados de esta combinación de n_ensemble
        row = {'n_ensemble': n}

        CM_list = subset_n['compl_measure'].unique()[1:] # to delete none

        for compl in CM_list:
            win, tie, loss = 0, 0, 0

            # Filtrar los datos para la medida de complejidad actual
            subset_compl = subset_n[subset_n['compl_measure'] == compl]
            #if (main_method == 'classic') or (compare_method == 'classic'):
            #     subset_compl_main = subset_n[subset_n['compl_measure'] == 'none']

            for dataset in subset_compl['dataset'].unique():
                # Filtrar para el dataset y métod en cuestión

                if (main_method == 'classic'):
                    subset_compl_main = subset_n[subset_n['compl_measure'] == 'none']
                    main_value = subset_compl_main[(subset_compl_main['dataset'] == dataset) & (subset_compl_main['method_weights'] == main_method)][metric].values
                else:
                    main_value = subset_compl[(subset_compl['dataset'] == dataset) & (subset_compl['method_weights'] == main_method)][metric].values
                if (compare_method == 'classic'):
                    subset_compl_compare = subset_n[subset_n['compl_measure'] == 'none']
                    compare_value = subset_compl_compare[(subset_compl_compare['dataset'] == dataset) & (subset_compl_compare['method_weights'] == compare_method)][metric].values
                else:
                    compare_value = subset_compl[(subset_compl['dataset'] == dataset) & (subset_compl['method_weights'] == compare_method)][metric].values

                # Verificar que tenemos valores únicos para cada dataset y métod
                if main_value.size > 0 and compare_value.size > 0:
                    main_value = main_value[0]
                    compare_value = compare_value[0]

                    # Comparación win-tie-loss
                    if main_value < compare_value:
                        win += 1
                        value_wtl = 'win'
                    elif main_value == compare_value:
                        tie += 1
                        value_wtl = 'tie'
                    else:
                        loss += 1
                        value_wtl = 'loss'

                # Obtener las medidas de complejidad para el dataset
                complexity_values = complexity_df[complexity_df['dataset'] == dataset][compl].values
                if complexity_values.size > 0:
                    complexity_value = complexity_values[0]

                    # Almacenar los resultados en una lista de diccionarios
                    all_results.append({
                        'n_ensemble': n,
                        'compl_measure': compl,
                        'result': value_wtl,
                        'complexity_value': complexity_value,
                        'dataset_name':dataset,
                        main_method:main_value,
                        compare_method:compare_value
                    })

            # Guardar el resultado para esta medida de complejidad en una columna específica
            row[f'{compare_method}_{compl}'] = (win, tie, loss)

        # Agregar el resultado de esta iteración a los resultados
        results.append(row)

    # Convertir los resultados a DataFrame
    win_tie_loss_df = pd.DataFrame(results)

    return win_tie_loss_df, all_results



def summaryStatistics_HardGB_perCM(df_hardGB2,CM):
    all_datasets_CM = df_hardGB2.loc[((df_hardGB2.compl_measure == CM) | (df_hardGB2.compl_measure == 'none')),:]


    summary_results_low = all_datasets_CM.groupby(['dataset', 'method_weights']).agg(
            test_acc_mean_median=('test_acc_mean', 'median'),
            test_acc_mean_std=('test_acc_mean', 'std'),
            test_acc_mean_total_mean=('test_acc_mean', 'mean')
        ).reset_index()
    summary_results_low

    # Pivotar para obtener una tabla más organizada
    summary_pivot_low = summary_results_low.pivot(index='dataset', columns='method_weights',
                                              values=['test_acc_mean_median', 'test_acc_mean_std', 'test_acc_mean_total_mean'])

    # Renombrar columnas para que sean más fáciles de leer
    summary_pivot_low.columns = [f'{method}_{stat}' for stat, method in summary_pivot_low.columns]
    summary_pivot_low.reset_index(inplace=True)

    # Plots
    # Low complexity dataset
    summary_long_low = summary_pivot_low.melt(id_vars='dataset',
                                             value_vars=[
                                                'classic_test_acc_mean_median',
                                                'sample_weight_easy_test_acc_mean_median',
                                                'classic_test_acc_mean_std',
                                                'sample_weight_easy_test_acc_mean_std',
                                                'classic_test_acc_mean_total_mean',
                                                'sample_weight_easy_test_acc_mean_total_mean',
                                            ],
                                            var_name='method_stat',
                                            value_name='value')

    # Separar los nombres de métod y estadístico
    summary_long_low[['method', 'statistic']] = summary_long_low['method_stat'].str.rsplit('_', n=1, expand=True)
    summary_long_low['method'] = summary_long_low['method'].replace({
             'classic_test_acc_mean': 'Classic',
            'sample_weight_easy_test_acc_mean': 'Easy',
            'classic_test_acc_mean_total': 'Classic',
            'sample_weight_easy_test_acc_mean_total': 'Easy',
    })


    # Crear un boxplot para cada estadístico
    plt.figure(figsize=(12, 5))

    # Usar un bucle para crear un boxplot por cada estadístico
    for i, stat in enumerate(summary_long_low['statistic'].unique(), start=1):
        plt.subplot(1, 3, i)  # 3 filas, 1 columna
        sns.boxplot(data=summary_long_low[summary_long_low['statistic'] == stat],
                         x='method',
                         y='value',
                         palette=colour_palette_personalized)

        plt.xlabel('Method')
        plt.ylabel(stat.capitalize())

    # Título global para el gráfico
    plt.suptitle('Boxplots of Summary Statistics Low accuracy Datasets '+CM, fontsize=16)


    plt.tight_layout()
    plt.show()
    return


### tod lo de arriba lo hemos copiado para tenerlo ahí

#############################################################################################################
#################                         QUIÉN SE DEGRADA MÁS?????                         #################
#############################################################################################################

## Como nuestra std es menor, vamos a ver si nosotros nos degradamos menos


#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import linregress
#
# performance_my_method = all_datasets.loc[(all_datasets['dataset']=='pyrim')&
#                  (all_datasets['loss_selected']=='exponential')&
#                 (all_datasets['method_weights']=='sample_weight_easy')&
#                     (all_datasets['compl_measure']=='N2'),'test_acc_mean']
#
# performance_classic = all_datasets.loc[(all_datasets['dataset']=='pyrim')&
#                  (all_datasets['loss_selected']=='exponential')&
#                 (all_datasets['method_weights']=='classic')&
#                     (all_datasets['compl_measure']=='none'),'test_acc_mean']
#
# # Suponiendo que performance_my_method y performance_classic son listas con los valores
# # de rendimiento en cada número de ensamblados.
#
# num_ensemblados = np.array([10, 25, 50, 100, 150, 200, 250, 300])  # Modifica si tienes otros valores
# num_ensemblados = range(300)
#
# # Regresión lineal para ver tendencia
# slope_my, intercept_my, r_value_my, p_value_my, _ = linregress(num_ensemblados, performance_my_method)
# slope_classic, intercept_classic, r_value_classic, p_value_classic, _ = linregress(num_ensemblados, performance_classic)
#
# # Visualización
# plt.figure(figsize=(8, 5))
# sns.lineplot(x=num_ensemblados, y=performance_my_method, marker='o', label="Métod Modificado")
# sns.lineplot(x=num_ensemblados, y=performance_classic, marker='s', linestyle="dashed", label="Gradient Boosting Clásico")
# plt.xlabel("Número de ensamblados")
# plt.ylabel("Rendimiento (accuracy, AUC, etc.)")
# plt.title("Degradación del rendimiento con más ensamblados")
# plt.legend()
# plt.show()
#
# # Interpretación
# print(f"Pendiente en mi métod: {slope_my:.4f} (p-valor: {p_value_my:.4f})")
# print(f"Pendiente en boosting clásico: {slope_classic:.4f} (p-valor: {p_value_classic:.4f})")
#
# if slope_my < slope_classic:
#     print("Mi métod se degrada menos con más ensamblados.")
# else:
#     print("El boosting clásico mantiene mejor el rendimiento con más ensamblados.")


from scipy.stats import linregress, wilcoxon
df_hardGB2
# Lista de todos los datasets
datasets = df_hardGB2['dataset'].unique()

# Para guardar performance
performance_CDGB = {}
performance_classic = {}

CM = 'Hostility'
# Iteramos sobre cada dataset para obtener su rendimiento
for dataset in datasets:
    performance_CDGB[dataset] = df_hardGB2.loc[
        (df_hardGB2['dataset'] == dataset) &
        (df_hardGB2['method_weights'] == 'sample_weight_easy') &
        (df_hardGB2['compl_measure'] == CM),
        'test_acc_mean'
    ].values
    performance_classic[dataset] = df_hardGB2.loc[
        (df_hardGB2['dataset'] == dataset) &
        (df_hardGB2['method_weights'] == 'classic') &
        (df_hardGB2['compl_measure'] == 'none'),
        'test_acc_mean'
    ].values


num_ensembles = range(300)

# Guardamos las pendientes para cada dataset
slopes_CDGB = []
slopes_classic = []

for dataset in performance_CDGB.keys():
    # Obtener rendimiento
    y_CDGB = performance_CDGB[dataset]
    y_classic = performance_classic[dataset]

    # Ajuste de regresión lineal
    slope_CDGB, _, _, p_CDGB, _ = linregress(num_ensembles, y_CDGB)
    slope_classic, _, _, p_classic, _ = linregress(num_ensembles, y_classic)

    slopes_CDGB.append(slope_CDGB)
    slopes_classic.append(slope_classic)

# Convertimos en DataFrame para análisis
df_slopes = pd.DataFrame({
    "Dataset": list(performance_CDGB.keys()),
    "Slopes CDGB": slopes_CDGB,
    "Slopes Classic": slopes_classic
})

# Visualización de la distribución de pendientes
plt.figure(figsize=(8, 5))
sns.kdeplot(slopes_CDGB, label="Complexity driven GB", fill=True)
sns.kdeplot(slopes_classic, label="Classic GB", fill=True, linestyle="dashed")
plt.axvline(np.mean(slopes_CDGB), color='blue', linestyle='dotted', label="Avg Complexity driven GB")
plt.axvline(np.mean(slopes_classic), color='red', linestyle='dotted', label="Avg Classic GB")
plt.xlabel("Regression slopes (degradación del rendimiento)")
plt.ylabel("Density")
plt.legend()
plt.title("Comparison")
plt.show()

# Test estadístico para comparar las pendientes
stat, p_value = wilcoxon(slopes_CDGB, slopes_classic)
p_value
# if p-value<0.05, hay diferencias significativas
# else, no hay diferencias significativas en la degradación del modelo

## FALTA HACERLO FUNCIÓN PARA LAS DISTINTAS MEDIDAS DE COMPLEJIDAD










#################################################################################################
#############                PARA OBTENER MÁS MEDIDAS DE EVALUACIÓN                 #############
#################################################################################################
# OJO! ESTOY HAY QUE HACERLO EN LOS FOLDS, NO EN LA VERSIÓN AGREGADA PORQUE NO SALE LO MISMO
# pero aquí tenemos el código para cuando lo hagamos

all_datasets.conf_matr_test_total[0]

import ast  # Para convertir la cadena en una lista de listas

# Convertimos la columna de matrices de confusión a listas de listas para que sea un objeto de python
all_datasets['conf_matr_test_total'] = all_datasets['conf_matr_test_total'].apply(ast.literal_eval)

all_datasets['TN'] = all_datasets['conf_matr_test_total'].apply(lambda x: x[0][0])
all_datasets['FP'] = all_datasets['conf_matr_test_total'].apply(lambda x: x[0][1])
all_datasets['FN'] = all_datasets['conf_matr_test_total'].apply(lambda x: x[1][0])
all_datasets['TP'] = all_datasets['conf_matr_test_total'].apply(lambda x: x[1][1])

# Calcular métricas de rendimiento
all_datasets['Accuracy'] = (all_datasets['TP'] + all_datasets['TN']) / (all_datasets['TP'] + all_datasets['TN'] + all_datasets['FP'] + all_datasets['FN'])
all_datasets['Precision'] = all_datasets['TP'] / (all_datasets['TP'] + all_datasets['FP'])
all_datasets['Recall'] = all_datasets['TP'] / (all_datasets['TP'] + all_datasets['FN'])
all_datasets['Specificity'] = all_datasets['TN'] / (all_datasets['TN'] + all_datasets['FP'])
all_datasets['F1_score'] = 2 * (all_datasets['Precision'] * all_datasets['Recall']) / (all_datasets['Precision'] + all_datasets['Recall'])
# Calcular MCC
all_datasets['MCC'] = (all_datasets['TP'] * all_datasets['TN'] - all_datasets['FP'] * all_datasets['FN']) / np.sqrt(
    (all_datasets['TP'] + all_datasets['FP']) * (all_datasets['TP'] + all_datasets['FN']) * (all_datasets['TN'] + all_datasets['FP']) * (all_datasets['TN'] + all_datasets['FN'])
)
# Reescalamos MCC de [-1, 1] a [0, 1]
all_datasets['MCC_scaled'] = (all_datasets['MCC'] + 1) / 2