# Scritp para unir los resultados de boosting obtenidos con factor 0.025 y factor 0.05

import pandas as pd
import glob
import re
import os

root_path = os.getcwd()

###################################################################################################################
####--------------                             RESULTADOS PARA FACTOR                            --------------####
###################################################################################################################

ruta_csvs = root_path + '/Results_Boostin_weights/*.csv'

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



###################################################################################################################
####--------------                   RESULTADOS UNIENDO TODAS LAS TIPOLOGÍAS                     --------------####
###################################################################################################################
# unimos classic, init_easy, init_hard, init_easy_x2, init_hard_x2, error_w_easy, error_w_hard


# Rutas de las dos carpetas
ruta_csvs_carpeta1 =  root_path + '/Results_Boostin_weights/*.csv'
ruta_csvs_carpeta2 =  root_path + '/Results_Boostin_x2/*.csv'

# Diccionarios para almacenar los datos combinados por dataset y tipo (totales o agregados)
datasets_total = {}
datasets_aggregated = {}



# Procesar los archivos de la primera carpeta
for file in glob.glob(ruta_csvs_carpeta1):
    # Extraer el nombre del dataset del archivo
    name_dataset = re.search('Boosting_(.*).csv', file).group(1)
    print(name_dataset)

    # Leer el CSV
    df = pd.read_csv(file)

    # Agregar la columna 'factor' si es necesario
    if 'Factor' not in df.columns:
        df['Factor'] = None
    #df = agregar_factor(df)

    # Verificar si es un archivo "total" o "agregado"
    if "Aggregated" in file:
        if name_dataset in datasets_aggregated:
            df = df[df['method_weights'] != 'classic']  # Filtrar para evitar duplicados de "classic"
            datasets_aggregated[name_dataset] = pd.concat([datasets_aggregated[name_dataset], df])
        else:
            datasets_aggregated[name_dataset] = df
    else:
        if name_dataset in datasets_total:
            df = df[df['method_weights'] != 'classic']  # Filtrar para evitar duplicados de "classic"
            datasets_total[name_dataset] = pd.concat([datasets_total[name_dataset], df])
        else:
            datasets_total[name_dataset] = df

# Procesar los archivos de la segunda carpeta
for file in glob.glob(ruta_csvs_carpeta2):
    # Extraer el nombre del dataset del archivo
    name_dataset = re.search('Boosting_(.*).csv', file).group(1)

    # Leer el CSV
    df = pd.read_csv(file)

    # Agregar la columna 'factor' si es necesario
    if 'Factor' not in df.columns:
        df['Factor'] = None
    #df = agregar_factor(df)

    # Verificar si es un archivo "total" o "agregado"
    if "Aggregated" in file:
        if name_dataset in datasets_aggregated:
            df = df[df['method_weights'] != 'classic']  # Filtrar para evitar duplicados de "classic"
            datasets_aggregated[name_dataset] = pd.concat([datasets_aggregated[name_dataset], df])
        else:
            datasets_aggregated[name_dataset] = df
    else:
        if name_dataset in datasets_total:
            df = df[df['method_weights'] != 'classic']  # Filtrar para evitar duplicados de "classic"
            datasets_total[name_dataset] = pd.concat([datasets_total[name_dataset], df])
        else:
            datasets_total[name_dataset] = df


# Guardar cada dataset combinado en un archivo CSV
for name_dataset, df_combined in datasets_total.items():
    file_out = f"Results_Boosting_{name_dataset}.csv"
    df_combined.to_csv(file_out, index=False)

for name_dataset, df_combined in datasets_aggregated.items():
    file_out = f"AggregatedResults_Boosting_{name_dataset}.csv"
    df_combined.to_csv(file_out, index=False)


###################################################################################################################
####--------------                   ELIMINAMOS VARIABLE FACTOR                     --------------####
###################################################################################################################

path_in = root_path + '/Results_Boosting_allmethods/*.csv'
path_out = root_path + '/Results_Boosting_allmethods_modificados'

# Asegurarse de que la carpeta de salida existe
os.makedirs(path_out, exist_ok=True)

for archivo in glob.glob(path_in):
    df = pd.read_csv(archivo)

    # Cambiamos la columna 'method_weights' para incluir la información de 'factor'
    df['method_weights'] = df.apply(
        lambda row: f"{row['method_weights']}_{row['Factor']}" if row['method_weights'] in ['error_w_easy',
                                                                                            'error_w_hard'] else row[
            'method_weights'],
        axis=1
    )

    # Eliminar la columna 'factor'
    df = df.drop(columns=['Factor'])

    nombre_archivo = os.path.basename(archivo)
    df.to_csv(os.path.join(path_out, nombre_archivo), index=False)




