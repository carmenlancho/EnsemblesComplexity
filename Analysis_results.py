#################  SCRIPT TO ANALYZE RESULTS FROM BAGGING ####################


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





def plot_acc_ensemble(data,name):
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


def plot_acc_ensemble_averaged(data,name):
    plt.plot(data.loc[data.weights == 'Averaged_measures','n_ensemble'],
             data.loc[data.weights == 'Averaged_measures','accuracy_mean'], c = 'red', label = 'Averaged_measures')
    plt.plot(data.loc[data.weights == 'Uniform','n_ensemble'],
             data.loc[data.weights == 'Uniform','accuracy_mean'], c = 'k', label = 'Uniform')
    plt.legend(loc='lower center', bbox_to_anchor=(0.55, 0.0),
          ncol=4, fancybox=False, shadow=False)
    plt.title(name)
    plt.show()

    return



def plot_acc_ensemble_and_average(data,data2,name):
    plt.plot(data.loc[data.weights == 'CLD','n_ensemble'],
             data.loc[data.weights == 'CLD','accuracy_mean'], c = 'blue', label = 'CLD', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'DCP','n_ensemble'],
             data.loc[data.weights == 'DCP','accuracy_mean'], c = 'green', label = 'DCP', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'LSC','n_ensemble'],
             data.loc[data.weights == 'LSC','accuracy_mean'], c = 'lime', label = 'LSC', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'TD_U','n_ensemble'],
             data.loc[data.weights == 'TD_U','accuracy_mean'], c = 'orange', label = 'TD_U', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'N2','n_ensemble'],
             data.loc[data.weights == 'N2','accuracy_mean'], c = 'purple', label = 'N2', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'F1','n_ensemble'],
             data.loc[data.weights == 'F1','accuracy_mean'], c = 'cyan', label = 'F1', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'Uniform','n_ensemble'],
             data.loc[data.weights == 'Uniform','accuracy_mean'], c = 'k', label = 'Uniform')
    plt.plot(data.loc[data.weights == 'N1','n_ensemble'],
             data.loc[data.weights == 'N1','accuracy_mean'], c = 'pink', label = 'N1', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'kDN','n_ensemble'],
             data.loc[data.weights == 'kDN','accuracy_mean'], c = 'gold', label = 'kDN', alpha = 0.5)
    plt.plot(data.loc[data.weights == 'Hostility','n_ensemble'],
             data.loc[data.weights == 'Hostility','accuracy_mean'], c = 'crimson', label = 'Hostility', alpha = 0.5)
    plt.plot(data2.loc[data2.weights == 'Averaged_measures','n_ensemble'],
             data2.loc[data2.weights == 'Averaged_measures','accuracy_mean'], c = 'red', label = 'Averaged_measures')
    plt.legend(loc='lower center', bbox_to_anchor=(0.55, 0.0),
          ncol=4, fancybox=False, shadow=False)
    plt.title(name)
    plt.show()

    return

#######################################################################
#################    ANALYSIS PER COMPLEXITY MEASURE   #################
#######################################################################


path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'yes' not in filename and 'classes' not in filename
    and 'averaged' not in filename):
        total_name_list.append(filename)


# data_list = ['Data1_','Data2_','Data3_','Data4_','Data5_','Data6_','Data7_','Data8_',
#              'Data9_','Data10_','Data11_','Data12_','Data13_']

data_list = ['Data1_','Data2_','Data3_','Data4_','Data5_','Data6_','Data7_','Data8_',
              'Data9_','Data10_','Data11_','Data12_','Data13_',
                          'pima','arrhythmia_cfs','vertebral_column','diabetic_retinopathy','segment',
             'breast-w','ilpd','diabetes',
             'ionosphere','sonar','banknote_authentication','wdbc']

path_to_save = root_path+'/Analysis_results'
res_all = pd.DataFrame()
for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    res_total = pd.DataFrame()
    for file in list_match:
        print(list_match)
        os.chdir(root_path + '/Bagging_results')
        name = file[25:]
        data = pd.read_csv(file)
        # data_name = file[file.find('AggregatedResults_Bagging_') + len('AggregatedResults_Bagging_'):file.rfind(
        #     '_MoreWeight')]

        data_n = data[data['n_ensemble'].isin([15, 50, 100, 150, 199])]
        res = data_n[['n_ensemble', 'weights', 'accuracy_mean', 'accuracy_std']].sort_values(
            by=['weights', 'n_ensemble'])  # .T
        if ('split4' in name and 'extreme' not in name and 'classic' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split4', 'accuracy_std_split4']
        elif ('split2' in name and 'extreme' not in name and 'classic' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split2', 'accuracy_std_split2']
        elif ('split9' in name and 'extreme' not in name and 'classic' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split9', 'accuracy_std_split9']
        elif ('split4' in name and 'extreme' in name and 'classic' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split4_extreme', 'accuracy_std_split4_extreme']
        elif ('split2' in name and 'extreme' in name and 'classic' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split2_extreme', 'accuracy_std_split2_extreme']
        elif ('split9' in name and 'extreme' in name and 'classic' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split9_extreme', 'accuracy_std_split9_extreme']
        elif ('hard' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_hard', 'accuracy_std_hard']
        elif ('easy' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_easy', 'accuracy_std_easy']
        elif ('combo' in name and 'extreme' not in name and 'split' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_combo', 'accuracy_std_combo']
        elif ('combo' in name and 'extreme' in name and 'split' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_combo_extreme', 'accuracy_std_combo_extreme']
        elif ('split4' in name and 'extreme' not in name and 'classic' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split4_classic', 'accuracy_std_split4_classic']
        elif ('split2' in name and 'extreme' not in name and 'classic' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split2_classic', 'accuracy_std_split2_classic']
        elif ('split1' in name and 'extreme' not in name and 'classic' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split1_classic', 'accuracy_std_split1_classic']
        elif ('split4' in name and 'extreme' in name and 'classic' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split4_classic_extreme', 'accuracy_std_split4_classic_extreme']
        elif ('split2' in name and 'extreme' in name and 'classic' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split2_classic_extreme', 'accuracy_std_split2_classic_extreme']
        elif ('split1' in name and 'extreme' in name and 'classic' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_split1_classic_extreme', 'accuracy_std_split1_classic_extreme']
        # res['dataset'] = data_i

        res_total = pd.concat([res_total, res], axis=1)
        # print(res_total)
    res_total['dataset'] = data_i
    res_total = res_total.loc[:, ~res_total.columns.duplicated()]  # remove duplicate columns
    res_total = res_total.reindex(columns=['n_ensemble', 'weights','dataset',
                                           'accuracy_mean_easy', 'accuracy_std_easy',
                                           'accuracy_mean_hard', 'accuracy_std_hard',
                                           'accuracy_mean_combo', 'accuracy_std_combo',
                                           'accuracy_mean_combo_extreme', 'accuracy_std_combo_extreme',
                                           'accuracy_mean_split2', 'accuracy_std_split2',
                                           'accuracy_mean_split2_extreme', 'accuracy_std_split2_extreme',
                                           'accuracy_mean_split1_classic', 'accuracy_std_split1_classic',
                                           'accuracy_mean_split1_classic_extreme', 'accuracy_std_split1_classic_extreme',
                                           'accuracy_mean_split4', 'accuracy_std_split4',
                                           'accuracy_mean_split4_extreme', 'accuracy_std_split4_extreme',
                                           'accuracy_mean_split2_classic', 'accuracy_std_split2_classic',
                                           'accuracy_mean_split2_classic_extreme', 'accuracy_std_split2_classic_extreme',
                                           'accuracy_mean_split9', 'accuracy_std_split9',
                                           'accuracy_mean_split9_extreme', 'accuracy_std_split9_extreme',
                                        'accuracy_mean_split4_classic','accuracy_std_split4_classic',
                                        'accuracy_mean_split4_classic_extreme','accuracy_std_split4_classic_extreme'])

    res_all = pd.concat([res_all, res_total])

    # To save the results
    # os.chdir(path_to_save)
    # nombre_csv = 'ResAccuracy_Bagging_' + str(data_i) + 'Classic_RealData_NoStump.csv'
    # res_total.to_csv(nombre_csv, encoding='utf_8_sig', index=True)

res_all.reset_index(inplace=True)
res_all.drop(['index'],inplace=True,axis=1)

list_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']
# Classic Bagging info
classic_values = res_all[res_all.weights == 'Uniform']
classic_values_mean = classic_values.groupby('dataset', as_index=False).mean()
classic_values_mean['n_ensemble'] = 'average'
classic_values_mean['weights'] = 'Uniform'
classic_values = pd.concat([classic_values,classic_values_mean])
classic_values.sort_values(by = ['dataset','n_ensemble'],inplace=True)

for CM in list_measures:
    print(CM)
    CM_results = res_all[res_all.weights == CM]
    total_average_CM = CM_results.groupby('dataset', as_index=False).mean()
    total_average_CM['n_ensemble'] = 'average'
    total_average_CM['weights'] = CM
    CM_results = pd.concat([CM_results, total_average_CM])
    CM_results.sort_values(by=['dataset', 'n_ensemble'], inplace=True)

    CM_results_complete = pd.concat([CM_results, classic_values])
    CM_results_complete.sort_values(by=['dataset', 'weights', 'n_ensemble'], inplace=True)

    ## Automatic comparison
    # CM_results[['accuracy_mean_easy', 'accuracy_mean_hard']]
    # classic_values[['n_ensemble', 'dataset', 'accuracy_mean_easy']]
    bag_value = np.array(classic_values[['accuracy_mean_easy']]) # tomamos este como ejemplo
    filter_col = [col for col in CM_results if col.startswith('accuracy_mean')]
    diff_with_classic = CM_results[filter_col] - bag_value
    diff_with_classic[['n_ensemble','weights','dataset']] = CM_results[['n_ensemble','weights','dataset']]
    diff_with_classic = diff_with_classic.reindex(columns=[ 'n_ensemble', 'weights',
    'dataset',
        'accuracy_mean_easy', 'accuracy_mean_hard', 'accuracy_mean_combo',
    'accuracy_mean_combo_extreme', 'accuracy_mean_split2',
    'accuracy_mean_split2_extreme', 'accuracy_mean_split1_classic',
    'accuracy_mean_split1_classic_extreme', 'accuracy_mean_split4',
    'accuracy_mean_split4_extreme', 'accuracy_mean_split2_classic',
    'accuracy_mean_split2_classic_extreme', 'accuracy_mean_split9',
    'accuracy_mean_split9_extreme', 'accuracy_mean_split4_classic',
    'accuracy_mean_split4_classic_extreme'])



    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracyPerMeasure_Bagging_' + str(CM) + '.csv'
    CM_results_complete.to_csv(nombre_csv, encoding='utf_8_sig', index=True)
    nombre_csv2 = 'ResDifAccuracyPerMeasure_Bagging_' + str(CM) + '.csv'
    diff_with_classic.to_csv(nombre_csv2, encoding='utf_8_sig', index=True)

    ## Win tie loss
    wtl_df = diff_with_classic.copy()
    wtl_df[filter_col] = np.where(wtl_df[filter_col] >= 0, 1, 0)
    nombre_csv3 = 'ResWTLAccuracyPerMeasure_Bagging_' + str(CM) + '.csv'
    wtl_df.to_csv(nombre_csv3, encoding='utf_8_sig', index=True)





#
# df_long = pd.melt(diff_with_classic, id_vars=['n_ensemble', 'weights','dataset'],
#                       value_vars=['accuracy_mean_easy','accuracy_mean_hard',
#                                   'accuracy_mean_combo','accuracy_mean_combo_extreme',
#                                   'accuracy_mean_split2', 'accuracy_mean_split2_extreme',
#                                     'accuracy_mean_split1_classic','accuracy_mean_split1_classic_extreme',
#                                     'accuracy_mean_split4',
#                                      'accuracy_mean_split4_extreme','accuracy_mean_split2_classic',
#                                      'accuracy_mean_split2_classic_extreme','accuracy_mean_split9',
#                                      'accuracy_mean_split9_extreme','accuracy_mean_split4_classic',
#                                      'accuracy_mean_split4_classic_extreme'],
#                       value_name='Accuracy')
# # we do not want to plot the average
# df_long.drop(df_long[df_long.n_ensemble =='average'].index, inplace=True)
#
# ax = sns.boxplot(y=df_long["Accuracy"], x=df_long["variable"],
#                 color='white',
#             order=['accuracy_mean_easy','accuracy_mean_hard',
#                                   'accuracy_mean_combo','accuracy_mean_combo_extreme',
#                                   'accuracy_mean_split2', 'accuracy_mean_split2_extreme',
#                                     'accuracy_mean_split1_classic','accuracy_mean_split1_classic_extreme',
#                                     'accuracy_mean_split4',
#                                      'accuracy_mean_split4_extreme','accuracy_mean_split2_classic',
#                                      'accuracy_mean_split2_classic_extreme','accuracy_mean_split9',
#                                      'accuracy_mean_split9_extreme','accuracy_mean_split4_classic',
#                                      'accuracy_mean_split4_classic_extreme']
#             )
# # sns.stripplot(data=df_long, x="variable", y="Accuracy", dodge=True,
# #               order=['accuracy_mean_easy', 'accuracy_mean_hard',
# #                      'accuracy_mean_combo', 'accuracy_mean_combo_extreme',
# #                      'accuracy_mean_split2', 'accuracy_mean_split2_extreme',
# #                      'accuracy_mean_split1_classic', 'accuracy_mean_split1_classic_extreme',
# #                      'accuracy_mean_split4',
# #                      'accuracy_mean_split4_extreme', 'accuracy_mean_split2_classic',
# #                      'accuracy_mean_split2_classic_extreme', 'accuracy_mean_split9',
# #                      'accuracy_mean_split9_extreme', 'accuracy_mean_split4_classic',
# #                      'accuracy_mean_split4_classic_extreme']
# #               )
# ax.set_xticklabels(['Easy', 'Hard', 'Combo', 'Combo_X',
#                          'Split2', 'Split2_X','Cl_Split1_3', 'Cl_Split1_3_X',
#                       'Split4','Split4_X',
#                           'Cl_Split2_5', 'Cl_Split2_5_X',
#                         'Split9', 'Split9_X',
#                          'Cl_Split4_9', 'Cl_Split4_9_X'],
#                         rotation=60)
# plt.show()



#######################################################################
#################    More weight in hard instances WITH RANKING   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' not in filename and 'average' not in filename and 'classes' not in filename):
        total_name_list.append(filename)
total_name_list.sort()

total_name_list2 = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' not in filename and 'average' in filename and 'classes' not in filename):
        total_name_list2.append(filename)
total_name_list2.sort()



# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)

for i in range(len(total_name_list)):
    print(i)
    os.chdir(root_path + '/Bagging_results')
    file1 = total_name_list[i]
    file2 = total_name_list2[i]
    name = file1[25:32]
    data = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    plot_acc_ensemble_and_average(data,data2, name)


path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble'])#.T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)


#######################################################################
#################    More weight in easy instances WITH RANKING    #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' not in filename and 'averaged' not in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)

path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble'])#.T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)



#######################################################################
#################    More weight in hard instances WITH RANKING classes   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)




#######################################################################
#################    More weight in easy instances WITH RANKING  classes   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' not in filename and 'classes' in filename):
        total_name_list.append(filename)



for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)



path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble']).T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)




#######################################################################
#################    More weight in combo (hard, easy) instances WITH RANKING   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'combo' in filename and 'classes' not in filename and 'averaged' not in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)


path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble'])#.T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)




#######################################################################
#################    More weight in combo split instances WITH RANKING   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'combo' in filename and 'classes' not in filename and 'split9' in filename):
        total_name_list.append(filename)


# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)


path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble'])#.T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)



#######################################################################
#################    More weight in combo split instances WITH RANKING stump   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'combo' in filename and 'classes' not in filename
    and 'no' in filename and 'extreme' in filename):
        total_name_list.append(filename)


# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)


path_to_save = root_path+'/Analysis_results'
for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble'])#.T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)


data_list = ['Data1_','Data2','Data3','Data4','Data5','Data6','Data7','Data8',
             'Data9','Data10','Data11','Data12','Data13',
             'wdbc','pima','ionosphere']
path_to_save = root_path+'/Analysis_results'
data_i = 'Data1_'
for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    res_total = pd.DataFrame()
    for file in list_match:
        print(list_match)
        os.chdir(root_path + '/Bagging_results')
        name = file[25:]
        data = pd.read_csv(file)

        data_n = data[data['n_ensemble'].isin([15, 50, 100, 150, 199])]
        res = data_n[['n_ensemble', 'weights', 'accuracy_mean', 'accuracy_std']].sort_values(
            by=['weights', 'n_ensemble'])  # .T
        if ('split4' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split4','accuracy_std_split4']
        elif ('split2' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split2','accuracy_std_split2']
        elif ('split9' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split9','accuracy_std_split9']
        elif ('split' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_combo', 'accuracy_std_combo']
        res_total = pd.concat([res_total,res],axis=1)
        # print(res_total)
    res_total = res_total.loc[:,~res_total.columns.duplicated()] # remove duplicate columns
    res_total = res_total.reindex(columns=['n_ensemble', 'weights',
                                               'accuracy_mean_combo', 'accuracy_std_combo',
                                               'accuracy_mean_split2', 'accuracy_std_split2',
                                               'accuracy_mean_split4', 'accuracy_std_split4',
                                               'accuracy_mean_split9', 'accuracy_std_split9'])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging_' + str(data_i) + '_Combo_Splits249_NoStump_Extreme.csv'
    res_total.to_csv(nombre_csv, encoding='utf_8_sig', index=True)






for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:]
    data = pd.read_csv(file)

    data_n = data[data['n_ensemble'].isin([15,50,100,150,199])]
    res = data_n[['n_ensemble','weights','accuracy_mean','accuracy_std']].sort_values(by = ['weights', 'n_ensemble'])#.T
    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging' + name
    res.to_csv(nombre_csv, encoding='utf_8_sig',index=True)



#################### REAL CASES
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'yes' not in filename and 'classes' not in filename
     and 'classic' in filename and 'Data' in filename):
        total_name_list.append(filename)



# data_list = ['wdbc','pima','ionosphere']
data_list = ['wdbc','pima','ionosphere','ilpd','diabetes','segment','sonar']

path_to_save = root_path+'/Analysis_results'

for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    res_total = pd.DataFrame()
    for file in list_match:
        print(file)
        os.chdir(root_path + '/Bagging_results')
        name = file[25:]
        data = pd.read_csv(file)

        data_n = data[data['n_ensemble'].isin([15, 50, 100, 150, 199])]
        res = data_n[['n_ensemble', 'weights', 'accuracy_mean', 'accuracy_std']].sort_values(
            by=['weights', 'n_ensemble'])  # .T
        if ('split4' in name and 'extreme' not in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split4','accuracy_std_split4']
        elif ('split2' in name and 'extreme' not in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split2','accuracy_std_split2']
        elif ('split9' in name and 'extreme' not in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split9','accuracy_std_split9']
        elif ('split4' in name and 'extreme' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split4_extreme','accuracy_std_split4_extreme']
        elif ('split2' in name and 'extreme' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split2_extreme','accuracy_std_split2_extreme']
        elif ('split9' in name and 'extreme' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split9_extreme','accuracy_std_split9_extreme']
        elif ('hard' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_hard', 'accuracy_std_hard']
        elif ('easy' in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_easy', 'accuracy_std_easy']
        elif ('combo' in name and 'extreme' not in name and 'split' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_combo', 'accuracy_std_combo']
        elif ('combo' in name and 'extreme' in name and 'split' not in name):
            res.columns = ['n_ensemble', 'weights', 'accuracy_mean_combo_extreme', 'accuracy_std_combo_extreme']
        res_total = pd.concat([res_total,res],axis=1)
        # print(res_total)
    res_total = res_total.loc[:,~res_total.columns.duplicated()] # remove duplicate columns
    res_total = res_total.reindex(columns=['n_ensemble', 'weights',
                                           # 'accuracy_mean_easy', 'accuracy_std_easy',
                                           # 'accuracy_mean_hard', 'accuracy_std_hard',
                                               'accuracy_mean_combo', 'accuracy_std_combo',
                                           'accuracy_mean_combo_extreme', 'accuracy_std_combo_extreme',
                                               'accuracy_mean_split2', 'accuracy_std_split2',
                                           'accuracy_mean_split2_extreme', 'accuracy_std_split2_extreme',
                                               'accuracy_mean_split4', 'accuracy_std_split4',
                                           'accuracy_mean_split4_extreme', 'accuracy_std_split4_extreme',
                                               'accuracy_mean_split9', 'accuracy_std_split9',
                                           'accuracy_mean_split9_extreme', 'accuracy_std_split9_extreme'])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging_' + str(data_i) + 'Classes_Combo_Splits249_NoStump.csv'
    res_total.to_csv(nombre_csv, encoding='utf_8_sig', index=True)



#################### CLASSIC CASES
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'yes' not in filename and 'classes' not in filename
     and 'classic' in filename and 'Data' not in filename):
        total_name_list.append(filename)


# data_list = ['Data1_','Data2_','Data3_','Data4_','Data5_','Data6_','Data7_','Data8_',
#              'Data9_','Data10_','Data11_','Data12_','Data13_']

data_list = ['pima','arrhythmia_cfs','vertebral_column','diabetic_retinopathy','segment',
             'breast-w','ilpd','diabetes',
             'ionosphere','sonar','banknote_authentication','wdbc']

path_to_save = root_path+'/Analysis_results'

for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    res_total = pd.DataFrame()
    for file in list_match:
        print(file)
        os.chdir(root_path + '/Bagging_results')
        name = file[25:]
        data = pd.read_csv(file)

        data_n = data[data['n_ensemble'].isin([15, 50, 100, 150, 199])]
        res = data_n[['n_ensemble', 'weights', 'accuracy_mean', 'accuracy_std']].sort_values(
            by=['weights', 'n_ensemble'])  # .T
        if ('split4' in name and 'extreme' not in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split4','accuracy_std_split4']
        elif ('split2' in name and 'extreme' not in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split2','accuracy_std_split2']
        elif ('split1' in name and 'extreme' not in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split1','accuracy_std_split1']
        elif ('split4' in name and 'extreme' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split4_extreme','accuracy_std_split4_extreme']
        elif ('split2' in name and 'extreme' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split2_extreme','accuracy_std_split2_extreme']
        elif ('split1' in name and 'extreme' in name):
            res.columns = ['n_ensemble', 'weights','accuracy_mean_split1_extreme','accuracy_std_split1_extreme']

        res_total = pd.concat([res_total,res],axis=1)
        # print(res_total)
    res_total = res_total.loc[:,~res_total.columns.duplicated()] # remove duplicate columns
    res_total = res_total.reindex(columns=['n_ensemble', 'weights',
                                               'accuracy_mean_split1', 'accuracy_std_split1',
                                           'accuracy_mean_split2', 'accuracy_std_split2',
                                               'accuracy_mean_split4', 'accuracy_std_split4',
                                           'accuracy_mean_split1_extreme', 'accuracy_std_split1_extreme',
                                           'accuracy_mean_split2_extreme', 'accuracy_std_split2_extreme',
                                               'accuracy_mean_split4_extreme', 'accuracy_std_split4_extreme',])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'ResAccuracy_Bagging_' + str(data_i) + 'Classic_RealData_NoStump.csv'
    res_total.to_csv(nombre_csv, encoding='utf_8_sig', index=True)






#######################################################################
#################    More weight in combo (hard, easy) instances WITH RANKING classes   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'combo' in filename and 'classes' in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble(data, name)



#######################################################################
#################    More weight in hard instances WITH RANKING AVERAGED   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and 'averaged' in filename and 'classes' not in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble_averaged(data, name)



#######################################################################
#################    More weight in easy instances WITH RANKING AVERAGED   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and 'averaged' in filename and 'classes' not in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble_averaged(data, name)



#######################################################################
#################    More weight in combo (hard, easy) instances WITH RANKING   #################
#######################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'combo' in filename and 'averaged' in filename and 'classes' not in filename):
        total_name_list.append(filename)


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    name = file[25:32]
    data = pd.read_csv(file)
    plot_acc_ensemble_averaged(data, name)



#
# #######################################################################
# #################    More weight in hard instances WITH 1/n +   #################
# #######################################################################
# path_csv = os.chdir(root_path+'/Bagging_results')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' in filename and 'classes' not in filename):
#         total_name_list.append(filename)
#
#
# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)
#
#
#
# #######################################################################
# #################    More weight in easy instances WITH 1/n +     #################
# #######################################################################
# path_csv = os.chdir(root_path+'/Bagging_results')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' in filename  and 'classes' not in filename):
#         total_name_list.append(filename)
#
#
#
# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)
#
#


#
#
# #######################################################################
# #################    More weight in hard instances WITH 1/n + classes  #################
# #######################################################################
# path_csv = os.chdir(root_path+'/Bagging_results')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'hard' in filename and 'Aggregated' in filename and '1n' in filename and 'classes' in filename):
#         total_name_list.append(filename)
#
#
# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)
#
#
#
# #######################################################################
# #################    More weight in easy instances WITH 1/n + classes    #################
# #######################################################################
# path_csv = os.chdir(root_path+'/Bagging_results')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'easy' in filename and 'Aggregated' in filename and '1n' in filename  and 'classes' in filename):
#         total_name_list.append(filename)
#
#
#
# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)



#
# #######################################################################
# #################    More weight in frontier    #################
# #######################################################################
# # We exclude LSC because it generally offers complexity values higher than 0.9
# path_csv = os.chdir(root_path+'/Bagging_results')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if (filename.endswith('.csv') and 'frontier' in filename and 'Aggregated' in filename):
#         total_name_list.append(filename)
#
#
#
# for file in total_name_list:
#     os.chdir(root_path + '/Bagging_results')
#     print(file)
#     name = file[25:32]
#     data = pd.read_csv(file)
#     plot_acc_ensemble(data, name)
#
#







