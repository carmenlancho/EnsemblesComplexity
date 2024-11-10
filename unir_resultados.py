# Scritp para unir los resultados de boosting obtenidos con factor 0.025 y factor 0.05

import pandas as pd
import glob
import re
import os

root_path = os.getcwd()
# Ruta donde se encuentran los CSVs
ruta_csvs = root_path + '/Results_Boostin_weights/*.csv'

total_name_list = ['teaching_assistant_MH.csv','contraceptive_NL.csv','hill_valley_without_noise_traintest.csv',
 'glass0.csv','saheart.csv','breast-w.csv','contraceptive_LS.csv', 'yeast1.csv','ilpd.csv',
    'phoneme.csv','mammographic.csv','contraceptive_NS.csv','bupa.csv','Yeast_CYTvsNUC.csv','ring.csv','titanic.csv',
 'musk1.csv','spectfheart.csv','arrhythmia_cfs.csv','vertebral_column.csv','profb.csv','sonar.csv',
 'liver-disorders.csv','steel-plates-fault.csv','credit-g.csv','glass1.csv','breastcancer.csv',
'diabetes.csv', 'diabetic_retinopathy.csv', 'WineQualityRed_5vs6.csv','teaching_assistant_LM.csv',
'ionosphere.csv', 'bands.csv','wdbc.csv','sylvine.csv', 'teaching_assistant_LH.csv',
'vehicle2.csv', 'pima.csv','spambase.csv','fri_c0_250_50.csv','parkinsons.csv','bodyfat.csv',
 'banknote_authentication.csv','chatfield_4.csv']


datasets_total = {}
datasets_aggregated = {}

for file in glob.glob(ruta_csvs):
    name_dataset = re.search('Boosting_(.*)_factor', file).group(1)

    # Leer el CSV actual
    df = pd.read_csv(file)

    # Añadir una columna con el valor del factor, extraído del nombre del archivo
    if "factor0025" in file:
        df['Factor'] = 0.025
    elif "factor005" in file:
        df['Factor'] = 0.05

    # Factor none for classic Boosting results
    df.loc[df['method_weights'] == 'classic', 'Factor'] = None

    if "Aggregated" in file:
        # Filtrar para mantener solo un conjunto de datos "classic"
        if name_dataset in datasets_aggregated:
            df = df[df['method_weights'] != 'classic']
            datasets_aggregated[name_dataset] = pd.concat([datasets_aggregated[name_dataset], df])
        else:
            datasets_aggregated[name_dataset] = df
    else:
        # Filtrar para mantener solo un conjunto de datos "classic"
        if name_dataset in datasets_total:
            df = df[df['method_weights'] != 'classic']
            datasets_total[name_dataset] = pd.concat([datasets_total[name_dataset], df])
        else:
            datasets_total[name_dataset] = df


# Guardar cada dataset combinado en un archivo CSV
for name_dataset, df_combined in datasets_total.items():
    file_out = f"Results_Boosting_{name_dataset}_weights_factor.csv"
    df_combined.to_csv(file_out, index=False)

for name_dataset, df_combined in datasets_aggregated.items():
    file_out = f"AggregatedResults_Boosting_{name_dataset}_weights_factor.csv"
    df_combined.to_csv(file_out, index=False)