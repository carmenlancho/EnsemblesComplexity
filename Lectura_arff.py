import os
import pandas as pd
import arff



from scipy.io import arff

# # Ruta donde están los archivos ARFF
# input_folder =  root_path + '/datasets/nuevos_datos'
# output_folder = root_path + '/datasets/datos_arff'
#
# # code
# arff_file = arff.loadarff(input_folder+'/ar1.arff')
# df = pd.DataFrame(arff_file[0])
# # path_csv = os.chdir(root_path+'/datasets')

import os
import pandas as pd
from scipy.io import arff

os.getcwd()

root_path = os.getcwd()

# Ruta donde están los archivos ARFF
input_folder = root_path + '/datasets/nuevos_datos'
output_folder = root_path + '/datasets/datos_arff'

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Inicializar una lista para almacenar estadísticas
stats = []

# Procesar cada archivo ARFF
for file_name in os.listdir(input_folder):
    if file_name.endswith(".arff"):
        file_path = os.path.join(input_folder, file_name)

        # Leer archivo ARFF
        arff_file = arff.loadarff(file_path)
        df = pd.DataFrame(arff_file[0])

        # Convertir columnas de tipo bytes a str
        for col in df.select_dtypes([object]):  # Seleccionar solo columnas de tipo object
            if isinstance(df[col].iloc[0], bytes):  # Verificar si la columna contiene bytes
                df[col] = df[col].str.decode('utf-8')

        # Guardar como CSV
        output_csv = os.path.join(output_folder, file_name.replace(".arff", ".csv"))
        df.to_csv(output_csv, index=False)

        # Calcular estadísticas
        total_instances = len(df)
        total_features = len(df.columns)
        minority_count = (df['class'] == 'minority').sum()  # Comparar con str después de la decodificación
        minority_percentage = (minority_count / total_instances) * 100 if total_instances > 0 else 0

        # Guardar estadísticas
        stats.append({
            "File": file_name,
            "Total Instances": total_instances,
            "Total Features": total_features,
            "Minority Instances": minority_count,
            "Minority Percentage": minority_percentage
        })

# Crear DataFrame con estadísticas
stats_df = pd.DataFrame(stats)

# Guardar estadísticas en un archivo CSV
stats_csv = os.path.join(output_folder, "info_datasets.csv")
stats_df.to_csv(stats_csv, index=False)
