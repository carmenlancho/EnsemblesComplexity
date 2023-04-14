###### Complexity analysis of the artificial datasets

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_path = os.getcwd()
path_csv = os.chdir(root_path+'/results')


# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv') and filename.startswith('metadata'):
        total_name_list.append(filename)


cols_name = ['dataset','feature_CL', 'feature_CLD', 'feature_DCP', 'feature_F1',
       'feature_Harmfulness', 'feature_LSC', 'feature_LSR', 'feature_N1',
       'feature_N2', 'feature_TD_P', 'feature_TD_U', 'feature_Usefulness',
       'feature_kDN', 'algo_bagging', 'algo_gradient_boosting',
       'algo_logistic_regression', 'algo_mlp', 'algo_random_forest',
       'algo_svc_linear', 'algo_svc_rbf']
data_total = pd.DataFrame(columns=cols_name)

for data_file in total_name_list:
    print(data_file)
    file = data_file

    file_name = data_file[9:17]
    data = pd.read_csv(file,index_col=0)
    data['dataset'] = file_name
    data_total = pd.concat([data_total, data], ignore_index=True)

data_total
sns.boxplot(data=data_total, x="feature_CLD", y="dataset")
plt.show()


sns.boxplot(data=data_total, x="feature_DCP", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="feature_F1", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="feature_LSC", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="feature_N1", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="feature_N2", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="feature_TD_U", y="dataset")
plt.show()

sns.boxplot(data=data_total, x="feature_kDN", y="dataset")
plt.show()