import matplotlib.pyplot as plt

X_bootstrap, y_bootstrap,
data = pd.DataFrame(X_bootstrap, columns=['x1','x2'])
data['y'] = y_bootstrap

# data = pd.DataFrame(X_train, columns=['x1','x2'])
# data['y'] = y_train
# len(y_train)
# len(y_bootstrap)
# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
idx_2 = np.where(data.y == 2)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=30, c='k', marker="*", label='l')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()

## easy
X_bootstrap, y_bootstrap,
data_easy = pd.DataFrame(X_bootstrap, columns=['x1','x2'])
data_easy['y'] = y_bootstrap

# Plot
# For labels
labels = list(data_easy.index)
idx_1 = np.where(data_easy.y == 1)
idx_0 = np.where(data_easy.y == 0)
idx_2 = np.where(data_easy.y == 2)
plt.scatter(data_easy.iloc[idx_0].x1, data_easy.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data_easy.iloc[idx_1].x1, data_easy.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data_easy.iloc[idx_2].x1, data_easy.iloc[idx_2].x2, s=30, c='k', marker="*", label='l')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()

## hard
X_bootstrap, y_bootstrap,
data_hard = pd.DataFrame(X_bootstrap, columns=['x1','x2'])
data_hard['y'] = y_bootstrap

# Plot
# For labels
labels = list(data_hard.index)
idx_1 = np.where(data_hard.y == 1)
idx_0 = np.where(data_hard.y == 0)
idx_2 = np.where(data_hard.y == 2)
plt.scatter(data_hard.iloc[idx_0].x1, data_hard.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data_hard.iloc[idx_1].x1, data_hard.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data_hard.iloc[idx_2].x1, data_hard.iloc[idx_2].x2, s=30, c='k', marker="*", label='l')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()






#########################################################################################
################## PLOT ACCURACY FOR one artificial case   ###############
#########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


root_path = os.getcwd()




def plot_acc_ensemble1(data,name):
    plt.plot(data.loc[data.weights == 'CLD','n_ensemble'],
             data.loc[data.weights == 'CLD','accuracy_mean'], c = 'blue', label = 'CLD')
    plt.plot(data.loc[data.weights == 'DCP','n_ensemble'],
             data.loc[data.weights == 'DCP','accuracy_mean'], c = 'green', label = 'DCP')
    plt.plot(data.loc[data.weights == 'LSC','n_ensemble'],
             data.loc[data.weights == 'LSC','accuracy_mean'], c = 'lime', label = 'LSC')
    plt.plot(data.loc[data.weights == 'TD_U','n_ensemble'],
             data.loc[data.weights == 'TD_U','accuracy_mean'], c = 'orange', label = 'TD_U')
    plt.plot(data.loc[data.weights == 'N2','n_ensemble'],
             data.loc[data.weights == 'N2','accuracy_mean'], c = 'purple', label = 'N2')
    plt.plot(data.loc[data.weights == 'F1','n_ensemble'],
             data.loc[data.weights == 'F1','accuracy_mean'], c = 'cyan', label = 'F1')
    plt.plot(data.loc[data.weights == 'Uniform','n_ensemble'],
             data.loc[data.weights == 'Uniform','accuracy_mean'], c = 'k', label = 'Uniform')
    plt.plot(data.loc[data.weights == 'N1','n_ensemble'],
             data.loc[data.weights == 'N1','accuracy_mean'], c = 'pink', label = 'N1')
    plt.plot(data.loc[data.weights == 'kDN','n_ensemble'],
             data.loc[data.weights == 'kDN','accuracy_mean'], c = 'gold', label = 'kDN')
    plt.plot(data.loc[data.weights == 'Hostility','n_ensemble'],
             data.loc[data.weights == 'Hostility','accuracy_mean'], c = 'crimson', label = 'Hostility')
    plt.legend(loc='lower center', bbox_to_anchor=(0.55, 0.0),
          ncol=4, fancybox=False, shadow=False)
    plt.title(name)
    plt.show()

    return


path_csv = os.chdir(root_path+'/Bagging_results')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'classic' in filename
#             and 'Aggregated' in filename and 'extreme' not in filename and 'averaged' not in filename
#             and 'split2' in filename):
#         total_name_list.append(filename)
# total_name_list.sort()
#
# total_name_list_uniform = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'easy' in filename and 'averaged' not in filename
#             and 'Aggregated' in filename and 'classes' not in filename):
#         total_name_list_uniform.append(filename)
# total_name_list_uniform.sort()

total_name_list = ['AggregatedResults_Bagging_Data9_MoreWeight_easy_Instances.csv',
 'AggregatedResults_Bagging_Data9_MoreWeight_hard_Instances.csv',
 'AggregatedResults_Bagging_Data9_MoreWeight_combo_split_classic_split1_stump_noInstances.csv']

data_easy = pd.read_csv('AggregatedResults_Bagging_Data6_MoreWeight_easy_Instances.csv')
data_hard = pd.read_csv('AggregatedResults_Bagging_Data6_MoreWeight_hard_Instances.csv')
data_classic = pd.read_csv('AggregatedResults_Bagging_Data6_MoreWeight_combo_split_classic_split1_stump_noInstances.csv')

data_together = pd.DataFrame()
acc_bagg = data_easy.loc[data_easy.weights == 'Uniform', ['n_ensemble','accuracy_mean']]
acc_bagg.columns = ['n_ensemble', 'acc_bagging']
acc_bagg.reset_index(inplace=True)
acc_bagg.drop(['index'],inplace=True,axis=1)
acc_easy = data_easy.loc[data_easy.weights == 'Hostility', ['n_ensemble','accuracy_mean']]
acc_easy.columns = ['n_ensemble', 'acc_host_easy']
acc_easy.reset_index(inplace=True)
acc_easy.drop(['index'],inplace=True,axis=1)
acc_hard = data_hard.loc[data_hard.weights == 'Hostility', ['n_ensemble','accuracy_mean']]
acc_hard.columns = ['n_ensemble', 'acc_host_hard']
acc_hard.reset_index(inplace=True)
acc_hard.drop(['index'],inplace=True,axis=1)
acc_classic = data_classic.loc[data_classic.weights == 'Hostility', ['n_ensemble','accuracy_mean']]
acc_classic.columns = ['n_ensemble', 'acc_host_classic1']
acc_classic.reset_index(inplace=True)
acc_classic.drop(['index'],inplace=True,axis=1)

data_together = pd.concat([data_together,acc_bagg],axis=1)
data_together = pd.concat([data_together,acc_easy],axis=1)
data_together = pd.concat([data_together,acc_hard],axis=1)
data_together = pd.concat([data_together,acc_classic],axis=1)
data_together = data_together.loc[:,~data_together.columns.duplicated()].copy()

data_n = data_together[data_together['n_ensemble'].isin([1,10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                       110, 120, 130, 140, 150, 160,
                                       170, 180, 190, 199])]

plt.plot(data_n['n_ensemble'],
         data_n['acc_bagging'], c='k',linestyle = 'dashed',linewidth=2, label='Uniform')
plt.plot(data_n['n_ensemble'],
         data_n['acc_host_easy'], c='royalblue',linestyle = 'dashdot',linewidth=2, label='Easy')
plt.plot(data_n['n_ensemble'],
         data_n['acc_host_hard'], c='deepskyblue',linestyle = 'dotted',linewidth=3, label='Hard')
plt.plot(data_n['n_ensemble'],
         data_n['acc_host_classic1'], c='crimson',linewidth=2, label='Mix')
plt.legend(loc='lower right',
           ncol=1, fancybox=False, shadow=False)
# plt.title('Data13')
plt.xlabel('Ensemble size')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()




