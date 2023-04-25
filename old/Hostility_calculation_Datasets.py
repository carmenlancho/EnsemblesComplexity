############ Script to obtain the hostility measure of the Artificial datasets

from Hostility_measure_algorithm import hostility_measure


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_path = os.getcwd()
path_csv = os.chdir(root_path+'/datasets')


# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file

    file_name = data_file[0:-4]
    data = pd.read_csv(file)
    y = data['y'].to_numpy()
    X = data.iloc[:, 0:-1].to_numpy()
    sigma = 5
    delta = 0.5
    seed = 0
    k_min = 0
    host_instance, data_clusters, results, k_auto = hostility_measure(sigma, X, y, delta, k_min, seed=0)
    host_instances = host_instance[k_auto]
    host_df = pd.DataFrame(host_instances)
    host_df.columns = ['Hostility']
    os.chdir(root_path + '/results')
    df_name = 'Host_'+ file_name + '.csv'
    host_df.to_csv(df_name)

