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

data_list = ['pima','arrhythmia_cfs','vertebral_column','diabetic_retinopathy','segment',
             'breast-w','ilpd','diabetes',
             'ionosphere','sonar','banknote_authentication','wdbc',
             'bands','bupa','contraceptive_LS','contraceptive_NL','contraceptive_NS',
             'credit-g','hill_valley_without_noise_traintest','mammographic',
             'phoneme','spambase','teaching_assistant_LH','teaching_assistant_LM','teaching_assistant_MH',
             'titanic','WineQualityRed_5vs6','Yeast_CYTvsNUC',
             'Data1_', 'Data2_', 'Data3_', 'Data4_', 'Data5_', 'Data6_', 'Data7_', 'Data8_',
            'Data9_','Data10_','Data11_','Data12_','Data13_']

path_to_save = root_path+'/Analysis_results_ranking_avg'
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

        data_n = data[data['n_ensemble'].isin([9, 19, 29,39,49,59,69,79,89,99,
                                               109,119,129,139,149,159,169,179,189, 199])]
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
    CM_results_complete.reset_index(inplace=True)
    CM_results_complete.drop('index',axis=1,inplace=True)
    sort_dict = {'bupa':1,'hill_valley_without_noise_traintest':2,'contraceptive_NS':3,
                 'teaching_assistant_LM':4,'contraceptive_LS':5,'diabetic_retinopathy':6,
                 'Yeast_CYTvsNUC':7,'bands':8,'ilpd':9,'teaching_assistant_LH':10,'teaching_assistant_MH':11,
                 'contraceptive_NL':12,'WineQualityRed_5vs6':13,'vertebral_column':14,
                 'diabetes':15,'credit-g':16,'arrhythmia_cfs':17,'pima':18,'mammographic':19,
                 'titanic':20,'sonar':21,'phoneme':22,'spambase':23, 'ionosphere':24,
                 'wdbc':25, 'segment':26,'breast-w':27,  'banknote_authentication':28,
                 'Data3_':29,'Data1_':30,'Data11_':31,'Data5_':32,'Data13_':33,
                 'Data9_':34,'Data2_':35, 'Data10_':36,'Data8_':37, 'Data6_':38,
                 'Data7_':39, 'Data12_':40, 'Data4_':41}
    sort_dict2 = {'15':1,'50':2,'100':3,'150':4,'199':5,'average':6}
    sort_dict3 = {'Uniform': 1}
    order = np.lexsort([CM_results_complete['n_ensemble'].map(sort_dict2),
                        CM_results_complete['weights'].map(sort_dict3),
                        CM_results_complete['dataset'].map(sort_dict)])
    CM_results_complete = CM_results_complete.iloc[order]
    nombre_csv = 'ResAccuracyPerMeasure_Bagging_10_' + str(CM) + '.csv'
    CM_results_complete.to_csv(nombre_csv, encoding='utf_8_sig', index=True)

    diff_with_classic.reset_index(inplace=True)
    diff_with_classic.drop('index',axis=1,inplace=True)
    order2 = np.lexsort([diff_with_classic['n_ensemble'].map(sort_dict2),
                        diff_with_classic['weights'].map(sort_dict3),
                        diff_with_classic['dataset'].map(sort_dict)])
    diff_with_classic = diff_with_classic.iloc[order2]
    nombre_csv2 = 'ResDifAccuracyPerMeasure_Bagging_10_' + str(CM) + '.csv'
    diff_with_classic.to_csv(nombre_csv2, encoding='utf_8_sig', index=True)

    ## Win tie loss
    wtl_df = diff_with_classic.copy()
    wtl_df[filter_col] = np.where(wtl_df[filter_col] >= 0, 1, 0)
    nombre_csv3 = 'ResWTLAccuracyPerMeasure_Bagging_10_' + str(CM) + '.csv'
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


#########################################################################################
################## PLOT ACCURACY FOR SPLIT1 CLASSIC AND SPLIT2 CLASSIC    ###############
#########################################################################################
path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'classic' in filename
            and 'Aggregated' in filename and 'extreme' not in filename and 'averaged' not in filename
            and 'split2' in filename):
        total_name_list.append(filename)
total_name_list.sort()

total_name_list_uniform = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'easy' in filename and 'averaged' not in filename
            and 'Aggregated' in filename and 'classes' not in filename):
        total_name_list_uniform.append(filename)
total_name_list_uniform.sort()

total_name_list = ['AggregatedResults_Bagging_Yeast_CYTvsNUC_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_bands_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_bupa_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_contraceptive_LS_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_contraceptive_NS_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_diabetic_retinopathy_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_hill_valley_without_noise_traintest_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_ilpd_MoreWeight_combo_split_classic_split2_stump_noInstances.csv',
 'AggregatedResults_Bagging_teaching_assistant_LM_MoreWeight_combo_split_classic_split2_stump_noInstances.csv']


for file in total_name_list:
    os.chdir(root_path + '/Bagging_results')
    print(file)
    # name = file[25:32]
    # name = 'Data1_'
    name = (file.split('AggregatedResults_Bagging_'))[1].split('MoreWeight')[0]
    matching = [s for s in total_name_list_uniform if name in s][0]
    data_unif = pd.read_csv(matching)
    data = pd.read_csv(file)
    data[data.weights =='Uniform'] = data_unif[data_unif.weights =='Uniform']
    data_n = data[data['n_ensemble'].isin([1,10,20,30,40,50,60,70,80,90,100,
                                           110,120,130,140,150,160,
                                           170,180,190,199])]
    plot_acc_ensemble(data_n, name)







#######################################################################
#################    ANALYSIS PER COMPLEXITY MEASURE ONLY FOR CLASSIC   #################
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

data_list = ['pima','arrhythmia_cfs','vertebral_column','diabetic_retinopathy','segment',
             'breast-w','ilpd','diabetes',
             'ionosphere','sonar','banknote_authentication','wdbc',
             'bands','bupa','contraceptive_LS','contraceptive_NL','contraceptive_NS',
             'credit-g','hill_valley_without_noise_traintest','mammographic',
             'phoneme','spambase','teaching_assistant_LH','teaching_assistant_LM','teaching_assistant_MH',
             'titanic','WineQualityRed_5vs6','Yeast_CYTvsNUC',
             'Data1_', 'Data2_', 'Data3_', 'Data4_', 'Data5_', 'Data6_', 'Data7_', 'Data8_',
            'Data9_','Data10_','Data11_','Data12_','Data13_']

path_to_save = root_path+'/Analysis_results_ranking_avg'
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

        # data_n = data[data['n_ensemble'].isin([15, 50, 100, 150, 199])]
        data = data[data.n_ensemble > 9]
        res = data[['n_ensemble', 'weights', 'accuracy_mean', 'accuracy_std']].sort_values(
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
classic_values_mean = classic_values.groupby('dataset', as_index=False).mean(numeric_only=True)
classic_values_mean['n_ensemble'] = 'average'
classic_values_mean['weights'] = 'Uniform'
classic_values = pd.concat([classic_values,classic_values_mean])
classic_values.sort_values(by = ['dataset','n_ensemble'],inplace=True)

total_mean_acc = pd.DataFrame()
total_mean_acc = pd.concat([total_mean_acc,classic_values_mean])
diff_classic = pd.DataFrame()
diff_classic_total = pd.DataFrame()
wtl_total = pd.DataFrame()

for CM in list_measures:
    print(CM)
    CM_results = res_all[res_all.weights == CM]
    total_average_CM = CM_results.groupby('dataset', as_index=False).mean(numeric_only=True)
    total_average_CM['n_ensemble'] = 'average'
    total_average_CM['weights'] = CM
    total_mean_acc = pd.concat([total_mean_acc,total_average_CM])
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
    diff_with_classic = diff_with_classic[diff_with_classic.n_ensemble != 'average']
    diff_classic_total = pd.concat([diff_classic_total, diff_with_classic])
    diff_with_classic_mean = diff_with_classic.groupby('dataset', as_index=False).mean(numeric_only=True)
    diff_with_classic_std = diff_with_classic.groupby('dataset', as_index=False).std(numeric_only=True)
    diff_with_classic_std.columns = ['dataset', 'accuracy_mean_easy_std', 'accuracy_mean_hard_std',
       'accuracy_mean_combo_std', 'accuracy_mean_combo_extreme_std',
       'accuracy_mean_split2_std', 'accuracy_mean_split2_extreme_std',
       'accuracy_mean_split1_classic_std', 'accuracy_mean_split1_classic_extreme_std',
       'accuracy_mean_split4_std', 'accuracy_mean_split4_extreme_std',
       'accuracy_mean_split2_classic_std', 'accuracy_mean_split2_classic_extreme_std',
       'accuracy_mean_split9_std', 'accuracy_mean_split9_extreme_std',
       'accuracy_mean_split4_classic_std', 'accuracy_mean_split4_classic_extreme_std']
    diff_with_classic_summary = pd.concat([diff_with_classic_mean,diff_with_classic_std], axis=1)
    diff_with_classic_summary = diff_with_classic_summary.loc[:, ~diff_with_classic_summary.columns.duplicated()]  # remove duplicate columns
    diff_with_classic_summary['weights'] = CM
    diff_classic = pd.concat([diff_classic, diff_with_classic_summary])




    # To save the results
    os.chdir(path_to_save)
    CM_results_complete.reset_index(inplace=True)
    CM_results_complete.drop('index',axis=1,inplace=True)
    sort_dict = {'bupa':1,'hill_valley_without_noise_traintest':2,'contraceptive_NS':3,
                 'teaching_assistant_LM':4,'contraceptive_LS':5,'diabetic_retinopathy':6,
                 'Yeast_CYTvsNUC':7,'bands':8,'ilpd':9,'teaching_assistant_LH':10,'teaching_assistant_MH':11,
                 'contraceptive_NL':12,'WineQualityRed_5vs6':13,'vertebral_column':14,
                 'diabetes':15,'credit-g':16,'arrhythmia_cfs':17,'pima':18,'mammographic':19,
                 'titanic':20,'sonar':21,'phoneme':22,'spambase':23, 'ionosphere':24,
                 'wdbc':25, 'segment':26,'breast-w':27,  'banknote_authentication':28,
                 'Data3_':29,'Data1_':30,'Data11_':31,'Data5_':32,'Data13_':33,
                 'Data9_':34,'Data2_':35, 'Data10_':36,'Data8_':37, 'Data6_':38,
                 'Data7_':39, 'Data12_':40, 'Data4_':41}
    sort_dict2 = {'15':1,'50':2,'100':3,'150':4,'199':5,'average':6}
    sort_dict3 = {'Uniform': 1}
    order = np.lexsort([CM_results_complete['n_ensemble'].map(sort_dict2),
                        CM_results_complete['weights'].map(sort_dict3),
                        CM_results_complete['dataset'].map(sort_dict)])
    CM_results_complete = CM_results_complete.iloc[order]
    # nombre_csv = 'ResAccuracyPerMeasure_Bagging_200_' + str(CM) + '.csv'
    # CM_results_complete.to_csv(nombre_csv, encoding='utf_8_sig', index=True)

    ## Win tie loss
    wtl_df = diff_with_classic.copy()
    wtl_df[filter_col] = np.where(wtl_df[filter_col] >= 0, 1, 0)
    wtl_df_mean = wtl_df.groupby(['dataset'], as_index=False).mean(numeric_only=True)
    wtl_df_mean['weights'] = CM
    wtl_total = pd.concat([wtl_total, wtl_df_mean])

order1 = np.lexsort([wtl_total['dataset'].map(sort_dict)])
wtl_total = wtl_total.iloc[order1]
nombre_csv3 = 'ResWTLAccuracy_Bagging_From10.csv'
wtl_total.to_csv(nombre_csv3, encoding='utf_8_sig', index=True)

order2 = np.lexsort([diff_classic['dataset'].map(sort_dict)])
diff_classic = diff_classic.iloc[order2]
nombre_csv2 = 'ResDifAccuracy_Bagging_From10.csv'
diff_classic.to_csv(nombre_csv2, encoding='utf_8_sig', index=True)

# Total mean and accuracy
order3 = np.lexsort([total_mean_acc['dataset'].map(sort_dict)])
total_mean_acc = total_mean_acc.iloc[order3]
nombre_csv4 = 'ResTotalSummaryMeansStd_Bagging_From10.csv'
total_mean_acc.to_csv(nombre_csv4, encoding='utf_8_sig', index=True)



## Boxplot from diff_classic_total
# Remove artificial dataset
diff_classic_total = diff_classic_total[~diff_classic_total.dataset.str.contains("Data")]
diff_classic_total = diff_classic_total[diff_classic_total.n_ensemble > 9]


diff_classic_total['complexity'] = 'Easy'
diff_classic_total
complex_datasets = ['bupa','hill_valley_without_noise_traintest','contraceptive_NS',
                 'teaching_assistant_LM','contraceptive_LS','diabetic_retinopathy',
                 'Yeast_CYTvsNUC','bands','ilpd']
Intermediate_datasets = ['teaching_assistant_LH','teaching_assistant_MH',
                 'contraceptive_NL','WineQualityRed_5vs6','vertebral_column',
                 'diabetes','credit-g','arrhythmia_cfs','pima','mammographic',
                 'titanic','sonar']
diff_classic_total['complexity'].loc[diff_classic_total['dataset'].isin(complex_datasets)] = 'Hard'
diff_classic_total['complexity'].loc[diff_classic_total['dataset'].isin(Intermediate_datasets)] = 'Intermediate'

hue_order = ['Easy','Intermediate','Hard']

plt.figure(figsize=(6.5,4.5))
# sns.color_palette("pastel")
ax = sns.boxplot(y=diff_classic_total["accuracy_mean_split1_classic_extreme"],
                 x=diff_classic_total["weights"], hue=diff_classic_total["complexity"],
                 hue_order=hue_order,palette="Blues",
            order=['Hostility',
                   'N1','N2','kDN','LSC','CLD' , 'TD_U','DCP','F1'])
            #      color='white')
ax.axhline(0, c='red')
# sns.stripplot(data=df_long_host, x="variable", y="Complexity", dodge=True, ax=ax,
#               order=['Uniform',
#                      'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
#                      'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
#                      'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
#                      'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
#                      'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme'])
ax.set_xticklabels(['Hostility','$N1_{HD}$','$N2_{HD}$',
                   'kDN','LSC','CLD' ,'TDU','DCP', '$F1_{HD}$'])
# ax.set_xticklabels(['Hostility','N1','N2',
#                    'kDN','LSC','CLD' ,'TDU','DCP', 'F1'])
ax.set(ylabel='Difference in accuracy', xlabel='')
# ax.legend(title='Datasets',ncol=1)
# sns.move_legend(ax, "lower center",
#     bbox_to_anchor=(.5, 1), ncol=3, title='Datasets')
ax.legend([],[], frameon=False)
plt.tight_layout()
plt.show()


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







