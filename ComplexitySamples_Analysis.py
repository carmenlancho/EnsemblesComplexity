################### SCRIPT TO ANALYZE COMPLEXITY OF THE BOOTSTRAP SAMPLES WITH EVERY STRATEGY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


root_path = os.getcwd()





path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'yes' not in filename and 'classes' not in filename
    and 'averaged' not in filename):
        total_name_list.append(filename)


###########################################################################################
##########                              DATASET LEVEL                          ############
###########################################################################################


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

# data_i = 'Data1_'

for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    data_total = pd.DataFrame()
    for file in list_match:
        print(list_match)
        os.chdir(root_path + '/Bagging_results')
        name = file[25:]
        data = pd.read_csv(file)
        if ('split4' in name and 'extreme' not in name and 'classic' not in name):
            data.columns = [str(col) + '_split4' for col in data.columns]
        elif ('split2' in name and 'extreme' not in name and 'classic' not in name):
            data.columns = [str(col) + '_split2' for col in data.columns]
        elif ('split9' in name and 'extreme' not in name and 'classic' not in name):
            data.columns = [str(col) + '_split9' for col in data.columns]
        elif ('split4' in name and 'extreme' in name and 'classic' not in name):
            data.columns = [str(col) + '_split4_extreme' for col in data.columns]
        elif ('split2' in name and 'extreme' in name and 'classic' not in name):
            data.columns = [str(col) + '_split2_extreme' for col in data.columns]
        elif ('split9' in name and 'extreme' in name and 'classic' not in name):
            data.columns = [str(col) + '_split9_extreme' for col in data.columns]
        elif ('hard' in name):
            data.columns = [str(col) + '_hard' for col in data.columns]
        elif ('easy' in name):
            data.columns = [str(col) + '_easy' for col in data.columns]
        elif ('combo' in name and 'extreme' not in name and 'split' not in name and 'classic' not in name):
            data.columns = [str(col) + '_combo' for col in data.columns]
        elif ('combo' in name and 'extreme' in name and 'split' not in name and 'classic' not in name):
            data.columns = [str(col) + '_combo_extreme' for col in data.columns]
        elif ('split1' in name and 'extreme' not in name and 'classic' in name):
            data.columns = [str(col) + '_split1_classic' for col in data.columns]
        elif ('split2' in name and 'extreme' not in name and 'classic' in name):
            data.columns = [str(col) + '_split2_classic' for col in data.columns]
        elif ('split4' in name and 'extreme' not in name and 'classic' in name):
            data.columns = [str(col) + '_split4_classic' for col in data.columns]
        elif ('split1' in name and 'extreme' in name and 'classic' in name):
            data.columns = [str(col) + '_split1_classic_extreme' for col in data.columns]
        elif ('split2' in name and 'extreme' in name and 'classic' in name):
            data.columns = [str(col) + '_split2_classic_extreme' for col in data.columns]
        elif ('split4' in name and 'extreme' in name and 'classic' in name):
            data.columns = [str(col) + '_split4_classic_extreme' for col in data.columns]
        data.columns.values[0:2] = ['n_ensemble', 'weights']
        data_total = pd.concat([data_total, data], axis=1)
    data_total = data_total.loc[:, ~data_total.columns.duplicated()]  # remove duplicate columns
    # data_total.columns

    # df_long_metric = pd.melt(data, id_vars=['dataset', 'resampling_method'],
    #                          value_vars=['IR_HostDifMean01_mean', 'IR_OvSize_mean'],
    #                          value_name='IR_Mean')
    # df_long_metric['IR_Mean'] = df_long_metric['IR_Mean'].astype('float')
    df_long_host = pd.melt(data_total[data_total['weights'] == 'Hostility'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
                                  'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
                                  'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
                                  'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
                                  'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme',
                                  'Boots_Hostility_dataset_mean_split1_classic', 'Boots_Hostility_dataset_mean_split1_classic_extreme',
                                  'Boots_Hostility_dataset_mean_split2_classic', 'Boots_Hostility_dataset_mean_split2_classic_extreme',
                                  'Boots_Hostility_dataset_mean_split4_classic', 'Boots_Hostility_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_kdn = pd.melt(data_total[data_total['weights'] == 'kDN'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_kDN_dataset_mean_easy', 'Boots_kDN_dataset_mean_hard',
                       'Boots_kDN_dataset_mean_combo', 'Boots_kDN_dataset_mean_combo_extreme',
                       'Boots_kDN_dataset_mean_split2', 'Boots_kDN_dataset_mean_split2_extreme',
                       'Boots_kDN_dataset_mean_split4', 'Boots_kDN_dataset_mean_split4_extreme',
                       'Boots_kDN_dataset_mean_split9', 'Boots_kDN_dataset_mean_split9_extreme',
                        'Boots_kDN_dataset_mean_split1_classic', 'Boots_kDN_dataset_mean_split1_classic_extreme',
                        'Boots_kDN_dataset_mean_split2_classic', 'Boots_kDN_dataset_mean_split2_classic_extreme',
                        'Boots_kDN_dataset_mean_split4_classic', 'Boots_kDN_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_dcp = pd.melt(data_total[data_total['weights'] == 'DCP'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_DCP_dataset_mean_easy', 'Boots_DCP_dataset_mean_hard',
                       'Boots_DCP_dataset_mean_combo', 'Boots_DCP_dataset_mean_combo_extreme',
                       'Boots_DCP_dataset_mean_split2', 'Boots_DCP_dataset_mean_split2_extreme',
                       'Boots_DCP_dataset_mean_split4', 'Boots_DCP_dataset_mean_split4_extreme',
                       'Boots_DCP_dataset_mean_split9', 'Boots_DCP_dataset_mean_split9_extreme',
                                  'Boots_DCP_dataset_mean_split1_classic', 'Boots_DCP_dataset_mean_split1_classic_extreme',
                                  'Boots_DCP_dataset_mean_split2_classic', 'Boots_DCP_dataset_mean_split2_classic_extreme',
                                  'Boots_DCP_dataset_mean_split4_classic', 'Boots_DCP_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_lsc = pd.melt(data_total[data_total['weights'] == 'LSC'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_LSC_dataset_mean_easy', 'Boots_LSC_dataset_mean_hard',
                       'Boots_LSC_dataset_mean_combo', 'Boots_LSC_dataset_mean_combo_extreme',
                       'Boots_LSC_dataset_mean_split2', 'Boots_LSC_dataset_mean_split2_extreme',
                       'Boots_LSC_dataset_mean_split4', 'Boots_LSC_dataset_mean_split4_extreme',
                       'Boots_LSC_dataset_mean_split9', 'Boots_LSC_dataset_mean_split9_extreme',
                                  'Boots_LSC_dataset_mean_split1_classic', 'Boots_LSC_dataset_mean_split1_classic_extreme',
                                  'Boots_LSC_dataset_mean_split2_classic', 'Boots_LSC_dataset_mean_split2_classic_extreme',
                                  'Boots_LSC_dataset_mean_split4_classic', 'Boots_LSC_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_n1 = pd.melt(data_total[data_total['weights'] == 'N1'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_N1_dataset_mean_easy', 'Boots_N1_dataset_mean_hard',
                       'Boots_N1_dataset_mean_combo', 'Boots_N1_dataset_mean_combo_extreme',
                       'Boots_N1_dataset_mean_split2', 'Boots_N1_dataset_mean_split2_extreme',
                       'Boots_N1_dataset_mean_split4', 'Boots_N1_dataset_mean_split4_extreme',
                       'Boots_N1_dataset_mean_split9', 'Boots_N1_dataset_mean_split9_extreme',
                                  'Boots_N1_dataset_mean_split1_classic', 'Boots_N1_dataset_mean_split1_classic_extreme',
                                  'Boots_N1_dataset_mean_split2_classic', 'Boots_N1_dataset_mean_split2_classic_extreme',
                                  'Boots_N1_dataset_mean_split4_classic', 'Boots_N1_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_n2 = pd.melt(data_total[data_total['weights'] == 'N2'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_N2_dataset_mean_easy', 'Boots_N2_dataset_mean_hard',
                       'Boots_N2_dataset_mean_combo', 'Boots_N2_dataset_mean_combo_extreme',
                       'Boots_N2_dataset_mean_split2', 'Boots_N2_dataset_mean_split2_extreme',
                       'Boots_N2_dataset_mean_split4', 'Boots_N2_dataset_mean_split4_extreme',
                       'Boots_N2_dataset_mean_split9', 'Boots_N2_dataset_mean_split9_extreme',
                                  'Boots_N2_dataset_mean_split1_classic', 'Boots_N2_dataset_mean_split1_classic_extreme',
                                  'Boots_N2_dataset_mean_split2_classic', 'Boots_N2_dataset_mean_split2_classic_extreme',
                                  'Boots_N2_dataset_mean_split4_classic', 'Boots_N2_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_f1 = pd.melt(data_total[data_total['weights'] == 'F1'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_F1_dataset_mean_easy', 'Boots_F1_dataset_mean_hard',
                         'Boots_F1_dataset_mean_combo', 'Boots_F1_dataset_mean_combo_extreme',
                         'Boots_F1_dataset_mean_split2', 'Boots_F1_dataset_mean_split2_extreme',
                         'Boots_F1_dataset_mean_split4', 'Boots_F1_dataset_mean_split4_extreme',
                         'Boots_F1_dataset_mean_split9', 'Boots_F1_dataset_mean_split9_extreme',
                                  'Boots_F1_dataset_mean_split1_classic', 'Boots_F1_dataset_mean_split1_classic_extreme',
                                  'Boots_F1_dataset_mean_split2_classic', 'Boots_F1_dataset_mean_split2_classic_extreme',
                                  'Boots_F1_dataset_mean_split4_classic', 'Boots_F1_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_tdu = pd.melt(data_total[data_total['weights'] == 'TD_U'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_TD_U_dataset_mean_easy', 'Boots_TD_U_dataset_mean_hard',
                       'Boots_TD_U_dataset_mean_combo', 'Boots_TD_U_dataset_mean_combo_extreme',
                       'Boots_TD_U_dataset_mean_split2', 'Boots_TD_U_dataset_mean_split2_extreme',
                       'Boots_TD_U_dataset_mean_split4', 'Boots_TD_U_dataset_mean_split4_extreme',
                       'Boots_TD_U_dataset_mean_split9', 'Boots_TD_U_dataset_mean_split9_extreme',
                                  'Boots_TD_U_dataset_mean_split1_classic', 'Boots_TD_U_dataset_mean_split1_classic_extreme',
                                  'Boots_TD_U_dataset_mean_split2_classic', 'Boots_TD_U_dataset_mean_split2_classic_extreme',
                                  'Boots_TD_U_dataset_mean_split4_classic', 'Boots_TD_U_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_cld = pd.melt(data_total[data_total['weights'] == 'CLD'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_CLD_dataset_mean_easy', 'Boots_CLD_dataset_mean_hard',
                         'Boots_CLD_dataset_mean_combo', 'Boots_CLD_dataset_mean_combo_extreme',
                         'Boots_CLD_dataset_mean_split2', 'Boots_CLD_dataset_mean_split2_extreme',
                         'Boots_CLD_dataset_mean_split4', 'Boots_CLD_dataset_mean_split4_extreme',
                         'Boots_CLD_dataset_mean_split9', 'Boots_CLD_dataset_mean_split9_extreme',
                                  'Boots_CLD_dataset_mean_split1_classic', 'Boots_CLD_dataset_mean_split1_classic_extreme',
                                  'Boots_CLD_dataset_mean_split2_classic', 'Boots_CLD_dataset_mean_split2_classic_extreme',
                                  'Boots_CLD_dataset_mean_split4_classic', 'Boots_CLD_dataset_mean_split4_classic_extreme'],
                      value_name='Complexity')


    data_aux_host = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_Hostility_dataset_mean_easy']]
    data_aux_host.columns = ['n_ensemble','weights','Complexity']
    data_aux_host['variable'] = 'Uniform'
    data_aux_kdn = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_kDN_dataset_mean_easy']]
    data_aux_kdn.columns = ['n_ensemble','weights','Complexity']
    data_aux_kdn['variable'] = 'Uniform'
    data_aux_dcp = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_DCP_dataset_mean_easy']]
    data_aux_dcp.columns = ['n_ensemble','weights','Complexity']
    data_aux_dcp['variable'] = 'Uniform'
    data_aux_lsc = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_LSC_dataset_mean_easy']]
    data_aux_lsc.columns = ['n_ensemble','weights','Complexity']
    data_aux_lsc['variable'] = 'Uniform'
    data_aux_n1 = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_N1_dataset_mean_easy']]
    data_aux_n1.columns = ['n_ensemble','weights','Complexity']
    data_aux_n1['variable'] = 'Uniform'
    data_aux_n2 = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_N2_dataset_mean_easy']]
    data_aux_n2.columns = ['n_ensemble','weights','Complexity']
    data_aux_n2['variable'] = 'Uniform'
    data_aux_f1 = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_F1_dataset_mean_easy']]
    data_aux_f1.columns = ['n_ensemble','weights','Complexity']
    data_aux_f1['variable'] = 'Uniform'
    data_aux_tdu = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_TD_U_dataset_mean_easy']]
    data_aux_tdu.columns = ['n_ensemble','weights','Complexity']
    data_aux_tdu['variable'] = 'Uniform'
    data_aux_cld = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_CLD_dataset_mean_easy']]
    data_aux_cld.columns = ['n_ensemble','weights','Complexity']
    data_aux_cld['variable'] = 'Uniform'
    df_long_host = pd.concat([df_long_host,data_aux_host],axis=0)
    df_long_kdn = pd.concat([df_long_kdn, data_aux_kdn], axis=0)
    df_long_dcp = pd.concat([df_long_dcp, data_aux_dcp], axis=0)
    df_long_lsc = pd.concat([df_long_lsc, data_aux_lsc], axis=0)
    df_long_n1 = pd.concat([df_long_n1, data_aux_n1], axis=0)
    df_long_n2 = pd.concat([df_long_n2, data_aux_n2], axis=0)
    df_long_f1 = pd.concat([df_long_f1, data_aux_f1], axis=0)
    df_long_tdu = pd.concat([df_long_tdu, data_aux_tdu], axis=0)
    df_long_cld = pd.concat([df_long_cld, data_aux_cld], axis=0)


    # Boxplot
    # ax = sns.boxplot(y=df_long_host["Complexity"], x=df_long_host["variable"],
    #             order=['Uniform',
    #                    'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
    #                               'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
    #                               'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
    #                               'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
    #                               'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme'],
    #                  color='white')
    # sns.stripplot(data=df_long_host, x="variable", y="Complexity", dodge=True, ax=ax,
    #               order=['Uniform',
    #                      'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
    #                      'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
    #                      'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
    #                      'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
    #                      'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme'])
    # ax.set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
    #                     'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X'],
    #                    rotation=40)
    # plt.show()

    fig, axes = plt.subplots(9, 1, figsize=(9, 50))
    sns.boxplot(ax=axes[0],y=df_long_host["Complexity"], x=df_long_host["variable"],
                order=['Uniform',
                       'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
                       'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
                       'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
                       'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
                       'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme',
                       'Boots_Hostility_dataset_mean_split1_classic',
                       'Boots_Hostility_dataset_mean_split1_classic_extreme',
                       'Boots_Hostility_dataset_mean_split2_classic',
                       'Boots_Hostility_dataset_mean_split2_classic_extreme',
                       'Boots_Hostility_dataset_mean_split4_classic',
                       'Boots_Hostility_dataset_mean_split4_classic_extreme'
                       ],
                color='white')
    sns.stripplot(data=df_long_host, x="variable", y="Complexity", dodge=True, ax=axes[0],
                  order=['Uniform',
                         'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
                         'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
                         'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
                         'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
                         'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme',
                         'Boots_Hostility_dataset_mean_split1_classic',
                         'Boots_Hostility_dataset_mean_split1_classic_extreme',
                         'Boots_Hostility_dataset_mean_split2_classic',
                         'Boots_Hostility_dataset_mean_split2_classic_extreme',
                         'Boots_Hostility_dataset_mean_split4_classic',
                         'Boots_Hostility_dataset_mean_split4_classic_extreme'
                         ])
    axes[0].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                       rotation=40)
    axes[0].set(ylabel='Hostility')
    sns.boxplot(ax=axes[1],y=df_long_kdn["Complexity"], x=df_long_kdn["variable"],
                order=['Uniform',
                       'Boots_kDN_dataset_mean_easy', 'Boots_kDN_dataset_mean_hard',
                       'Boots_kDN_dataset_mean_combo', 'Boots_kDN_dataset_mean_combo_extreme',
                       'Boots_kDN_dataset_mean_split2', 'Boots_kDN_dataset_mean_split2_extreme',
                       'Boots_kDN_dataset_mean_split4', 'Boots_kDN_dataset_mean_split4_extreme',
                       'Boots_kDN_dataset_mean_split9', 'Boots_kDN_dataset_mean_split9_extreme',
                       'Boots_kDN_dataset_mean_split1_classic', 'Boots_kDN_dataset_mean_split1_classic_extreme',
                       'Boots_kDN_dataset_mean_split2_classic', 'Boots_kDN_dataset_mean_split2_classic_extreme',
                       'Boots_kDN_dataset_mean_split4_classic', 'Boots_kDN_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_kdn, x="variable", y="Complexity", dodge=True, ax=axes[1],
                  order=['Uniform',
                         'Boots_kDN_dataset_mean_easy', 'Boots_kDN_dataset_mean_hard',
                         'Boots_kDN_dataset_mean_combo', 'Boots_kDN_dataset_mean_combo_extreme',
                         'Boots_kDN_dataset_mean_split2', 'Boots_kDN_dataset_mean_split2_extreme',
                         'Boots_kDN_dataset_mean_split4', 'Boots_kDN_dataset_mean_split4_extreme',
                         'Boots_kDN_dataset_mean_split9', 'Boots_kDN_dataset_mean_split9_extreme',
                         'Boots_kDN_dataset_mean_split1_classic', 'Boots_kDN_dataset_mean_split1_classic_extreme',
                        'Boots_kDN_dataset_mean_split2_classic', 'Boots_kDN_dataset_mean_split2_classic_extreme',
                        'Boots_kDN_dataset_mean_split4_classic', 'Boots_kDN_dataset_mean_split4_classic_extreme'])
    axes[1].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                       rotation=40)
    axes[1].set(ylabel='kDN')
    sns.boxplot(ax=axes[2], y=df_long_dcp["Complexity"], x=df_long_dcp["variable"],
                order=['Uniform',
                       'Boots_DCP_dataset_mean_easy', 'Boots_DCP_dataset_mean_hard',
                       'Boots_DCP_dataset_mean_combo', 'Boots_DCP_dataset_mean_combo_extreme',
                       'Boots_DCP_dataset_mean_split2', 'Boots_DCP_dataset_mean_split2_extreme',
                       'Boots_DCP_dataset_mean_split4', 'Boots_DCP_dataset_mean_split4_extreme',
                       'Boots_DCP_dataset_mean_split9', 'Boots_DCP_dataset_mean_split9_extreme',
                       'Boots_DCP_dataset_mean_split1_classic', 'Boots_DCP_dataset_mean_split1_classic_extreme',
                       'Boots_DCP_dataset_mean_split2_classic', 'Boots_DCP_dataset_mean_split2_classic_extreme',
                       'Boots_DCP_dataset_mean_split4_classic', 'Boots_DCP_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_dcp, x="variable", y="Complexity", dodge=True, ax=axes[2],
                  order=['Uniform',
                         'Boots_DCP_dataset_mean_easy', 'Boots_DCP_dataset_mean_hard',
                         'Boots_DCP_dataset_mean_combo', 'Boots_DCP_dataset_mean_combo_extreme',
                         'Boots_DCP_dataset_mean_split2', 'Boots_DCP_dataset_mean_split2_extreme',
                         'Boots_DCP_dataset_mean_split4', 'Boots_DCP_dataset_mean_split4_extreme',
                         'Boots_DCP_dataset_mean_split9', 'Boots_DCP_dataset_mean_split9_extreme',
                         'Boots_DCP_dataset_mean_split1_classic', 'Boots_DCP_dataset_mean_split1_classic_extreme',
                         'Boots_DCP_dataset_mean_split2_classic', 'Boots_DCP_dataset_mean_split2_classic_extreme',
                         'Boots_DCP_dataset_mean_split4_classic', 'Boots_DCP_dataset_mean_split4_classic_extreme'])
    axes[2].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[2].set(ylabel='DCP')
    sns.boxplot(ax=axes[3], y=df_long_lsc["Complexity"], x=df_long_lsc["variable"],
                order=['Uniform',
                       'Boots_LSC_dataset_mean_easy', 'Boots_LSC_dataset_mean_hard',
                       'Boots_LSC_dataset_mean_combo', 'Boots_LSC_dataset_mean_combo_extreme',
                       'Boots_LSC_dataset_mean_split2', 'Boots_LSC_dataset_mean_split2_extreme',
                       'Boots_LSC_dataset_mean_split4', 'Boots_LSC_dataset_mean_split4_extreme',
                       'Boots_LSC_dataset_mean_split9', 'Boots_LSC_dataset_mean_split9_extreme',
                       'Boots_LSC_dataset_mean_split1_classic', 'Boots_LSC_dataset_mean_split1_classic_extreme',
                       'Boots_LSC_dataset_mean_split2_classic', 'Boots_LSC_dataset_mean_split2_classic_extreme',
                       'Boots_LSC_dataset_mean_split4_classic', 'Boots_LSC_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_lsc, x="variable", y="Complexity", dodge=True, ax=axes[3],
                  order=['Uniform',
                         'Boots_LSC_dataset_mean_easy', 'Boots_LSC_dataset_mean_hard',
                         'Boots_LSC_dataset_mean_combo', 'Boots_LSC_dataset_mean_combo_extreme',
                         'Boots_LSC_dataset_mean_split2', 'Boots_LSC_dataset_mean_split2_extreme',
                         'Boots_LSC_dataset_mean_split4', 'Boots_LSC_dataset_mean_split4_extreme',
                         'Boots_LSC_dataset_mean_split9', 'Boots_LSC_dataset_mean_split9_extreme',
                         'Boots_LSC_dataset_mean_split1_classic', 'Boots_LSC_dataset_mean_split1_classic_extreme',
                         'Boots_LSC_dataset_mean_split2_classic', 'Boots_LSC_dataset_mean_split2_classic_extreme',
                         'Boots_LSC_dataset_mean_split4_classic', 'Boots_LSC_dataset_mean_split4_classic_extreme'])
    axes[3].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[3].set(ylabel='LSC')
    sns.boxplot(ax=axes[4], y=df_long_n1["Complexity"], x=df_long_n1["variable"],
                order=['Uniform',
                       'Boots_N1_dataset_mean_easy', 'Boots_N1_dataset_mean_hard',
                       'Boots_N1_dataset_mean_combo', 'Boots_N1_dataset_mean_combo_extreme',
                       'Boots_N1_dataset_mean_split2', 'Boots_N1_dataset_mean_split2_extreme',
                       'Boots_N1_dataset_mean_split4', 'Boots_N1_dataset_mean_split4_extreme',
                       'Boots_N1_dataset_mean_split9', 'Boots_N1_dataset_mean_split9_extreme',
                       'Boots_N1_dataset_mean_split1_classic', 'Boots_N1_dataset_mean_split1_classic_extreme',
                       'Boots_N1_dataset_mean_split2_classic', 'Boots_N1_dataset_mean_split2_classic_extreme',
                       'Boots_N1_dataset_mean_split4_classic', 'Boots_N1_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_n1, x="variable", y="Complexity", dodge=True, ax=axes[4],
                  order=['Uniform',
                         'Boots_N1_dataset_mean_easy', 'Boots_N1_dataset_mean_hard',
                         'Boots_N1_dataset_mean_combo', 'Boots_N1_dataset_mean_combo_extreme',
                         'Boots_N1_dataset_mean_split2', 'Boots_N1_dataset_mean_split2_extreme',
                         'Boots_N1_dataset_mean_split4', 'Boots_N1_dataset_mean_split4_extreme',
                         'Boots_N1_dataset_mean_split9', 'Boots_N1_dataset_mean_split9_extreme',
                         'Boots_N1_dataset_mean_split1_classic', 'Boots_N1_dataset_mean_split1_classic_extreme',
                         'Boots_N1_dataset_mean_split2_classic', 'Boots_N1_dataset_mean_split2_classic_extreme',
                         'Boots_N1_dataset_mean_split4_classic', 'Boots_N1_dataset_mean_split4_classic_extreme'])
    axes[4].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[4].set(ylabel='N1')
    sns.boxplot(ax=axes[5], y=df_long_n2["Complexity"], x=df_long_n2["variable"],
                order=['Uniform',
                       'Boots_N2_dataset_mean_easy', 'Boots_N2_dataset_mean_hard',
                       'Boots_N2_dataset_mean_combo', 'Boots_N2_dataset_mean_combo_extreme',
                       'Boots_N2_dataset_mean_split2', 'Boots_N2_dataset_mean_split2_extreme',
                       'Boots_N2_dataset_mean_split4', 'Boots_N2_dataset_mean_split4_extreme',
                       'Boots_N2_dataset_mean_split9', 'Boots_N2_dataset_mean_split9_extreme',
                       'Boots_N2_dataset_mean_split1_classic', 'Boots_N2_dataset_mean_split1_classic_extreme',
                       'Boots_N2_dataset_mean_split2_classic', 'Boots_N2_dataset_mean_split2_classic_extreme',
                       'Boots_N2_dataset_mean_split4_classic', 'Boots_N2_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_n2, x="variable", y="Complexity", dodge=True, ax=axes[5],
                  order=['Uniform',
                         'Boots_N2_dataset_mean_easy', 'Boots_N2_dataset_mean_hard',
                         'Boots_N2_dataset_mean_combo', 'Boots_N2_dataset_mean_combo_extreme',
                         'Boots_N2_dataset_mean_split2', 'Boots_N2_dataset_mean_split2_extreme',
                         'Boots_N2_dataset_mean_split4', 'Boots_N2_dataset_mean_split4_extreme',
                         'Boots_N2_dataset_mean_split9', 'Boots_N2_dataset_mean_split9_extreme',
                         'Boots_N2_dataset_mean_split1_classic', 'Boots_N2_dataset_mean_split1_classic_extreme',
                         'Boots_N2_dataset_mean_split2_classic', 'Boots_N2_dataset_mean_split2_classic_extreme',
                         'Boots_N2_dataset_mean_split4_classic', 'Boots_N2_dataset_mean_split4_classic_extreme'])
    axes[5].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[5].set(ylabel='N2')
    sns.boxplot(ax=axes[6], y=df_long_f1["Complexity"], x=df_long_f1["variable"],
                order=['Uniform',
                       'Boots_F1_dataset_mean_easy', 'Boots_F1_dataset_mean_hard',
                       'Boots_F1_dataset_mean_combo', 'Boots_F1_dataset_mean_combo_extreme',
                       'Boots_F1_dataset_mean_split2', 'Boots_F1_dataset_mean_split2_extreme',
                       'Boots_F1_dataset_mean_split4', 'Boots_F1_dataset_mean_split4_extreme',
                       'Boots_F1_dataset_mean_split9', 'Boots_F1_dataset_mean_split9_extreme',
                       'Boots_F1_dataset_mean_split1_classic', 'Boots_F1_dataset_mean_split1_classic_extreme',
                       'Boots_F1_dataset_mean_split2_classic', 'Boots_F1_dataset_mean_split2_classic_extreme',
                       'Boots_F1_dataset_mean_split4_classic', 'Boots_F1_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_f1, x="variable", y="Complexity", dodge=True, ax=axes[6],
                  order=['Uniform',
                         'Boots_F1_dataset_mean_easy', 'Boots_F1_dataset_mean_hard',
                         'Boots_F1_dataset_mean_combo', 'Boots_F1_dataset_mean_combo_extreme',
                         'Boots_F1_dataset_mean_split2', 'Boots_F1_dataset_mean_split2_extreme',
                         'Boots_F1_dataset_mean_split4', 'Boots_F1_dataset_mean_split4_extreme',
                         'Boots_F1_dataset_mean_split9', 'Boots_F1_dataset_mean_split9_extreme',
                         'Boots_F1_dataset_mean_split1_classic', 'Boots_F1_dataset_mean_split1_classic_extreme',
                         'Boots_F1_dataset_mean_split2_classic', 'Boots_F1_dataset_mean_split2_classic_extreme',
                         'Boots_F1_dataset_mean_split4_classic', 'Boots_F1_dataset_mean_split4_classic_extreme'])
    axes[6].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[6].set(ylabel='F1')
    sns.boxplot(ax=axes[7], y=df_long_tdu["Complexity"], x=df_long_tdu["variable"],
                order=['Uniform',
                       'Boots_TD_U_dataset_mean_easy', 'Boots_TD_U_dataset_mean_hard',
                       'Boots_TD_U_dataset_mean_combo', 'Boots_TD_U_dataset_mean_combo_extreme',
                       'Boots_TD_U_dataset_mean_split2', 'Boots_TD_U_dataset_mean_split2_extreme',
                       'Boots_TD_U_dataset_mean_split4', 'Boots_TD_U_dataset_mean_split4_extreme',
                       'Boots_TD_U_dataset_mean_split9', 'Boots_TD_U_dataset_mean_split9_extreme',
                       'Boots_TD_U_dataset_mean_split1_classic', 'Boots_TD_U_dataset_mean_split1_classic_extreme',
                       'Boots_TD_U_dataset_mean_split2_classic', 'Boots_TD_U_dataset_mean_split2_classic_extreme',
                       'Boots_TD_U_dataset_mean_split4_classic', 'Boots_TD_U_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_tdu, x="variable", y="Complexity", dodge=True, ax=axes[7],
                  order=['Uniform',
                         'Boots_TD_U_dataset_mean_easy', 'Boots_TD_U_dataset_mean_hard',
                         'Boots_TD_U_dataset_mean_combo', 'Boots_TD_U_dataset_mean_combo_extreme',
                         'Boots_TD_U_dataset_mean_split2', 'Boots_TD_U_dataset_mean_split2_extreme',
                         'Boots_TD_U_dataset_mean_split4', 'Boots_TD_U_dataset_mean_split4_extreme',
                         'Boots_TD_U_dataset_mean_split9', 'Boots_TD_U_dataset_mean_split9_extreme',
                         'Boots_TD_U_dataset_mean_split1_classic', 'Boots_TD_U_dataset_mean_split1_classic_extreme',
                         'Boots_TD_U_dataset_mean_split2_classic', 'Boots_TD_U_dataset_mean_split2_classic_extreme',
                         'Boots_TD_U_dataset_mean_split4_classic', 'Boots_TD_U_dataset_mean_split4_classic_extreme'])
    axes[7].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[7].set(ylabel='TDU')
    sns.boxplot(ax=axes[8], y=df_long_cld["Complexity"], x=df_long_cld["variable"],
                order=['Uniform',
                       'Boots_CLD_dataset_mean_easy', 'Boots_CLD_dataset_mean_hard',
                       'Boots_CLD_dataset_mean_combo', 'Boots_CLD_dataset_mean_combo_extreme',
                       'Boots_CLD_dataset_mean_split2', 'Boots_CLD_dataset_mean_split2_extreme',
                       'Boots_CLD_dataset_mean_split4', 'Boots_CLD_dataset_mean_split4_extreme',
                       'Boots_CLD_dataset_mean_split9', 'Boots_CLD_dataset_mean_split9_extreme',
                       'Boots_CLD_dataset_mean_split1_classic', 'Boots_CLD_dataset_mean_split1_classic_extreme',
                       'Boots_CLD_dataset_mean_split2_classic', 'Boots_CLD_dataset_mean_split2_classic_extreme',
                       'Boots_CLD_dataset_mean_split4_classic', 'Boots_CLD_dataset_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_cld, x="variable", y="Complexity", dodge=True, ax=axes[8],
                  order=['Uniform',
                         'Boots_CLD_dataset_mean_easy', 'Boots_CLD_dataset_mean_hard',
                         'Boots_CLD_dataset_mean_combo', 'Boots_CLD_dataset_mean_combo_extreme',
                         'Boots_CLD_dataset_mean_split2', 'Boots_CLD_dataset_mean_split2_extreme',
                         'Boots_CLD_dataset_mean_split4', 'Boots_CLD_dataset_mean_split4_extreme',
                         'Boots_CLD_dataset_mean_split9', 'Boots_CLD_dataset_mean_split9_extreme',
                         'Boots_CLD_dataset_mean_split1_classic', 'Boots_CLD_dataset_mean_split1_classic_extreme',
                         'Boots_CLD_dataset_mean_split2_classic', 'Boots_CLD_dataset_mean_split2_classic_extreme',
                         'Boots_CLD_dataset_mean_split4_classic', 'Boots_CLD_dataset_mean_split4_classic_extreme'])
    axes[8].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[8].set(ylabel='CLD')

    # plt.show()
    name_plot = 'ComplexityAnalysis_' + str(data_i) + '.png'
    os.chdir(path_to_save)
    plt.savefig(name_plot)
    plt.clf()



##############################################################################################
############                             CLASS LEVEL                          ################
##############################################################################################


path_csv = os.chdir(root_path+'/Bagging_results')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if (filename.endswith('.csv') and 'Aggregated' in filename and 'yes' not in filename and 'classes' not in filename
    and 'averaged' not in filename):
        total_name_list.append(filename)




data_list = ['Data1_','Data2','Data3','Data4','Data5','Data6','Data7','Data8',
             'Data9','Data10','Data11','Data12','Data13',
             'pima','segment','ilpd','diabetes','ionosphere','sonar','wdbc','vertebral_column',
            'diabetic_retinopathy', 'breast-w','arrhythmia_cfs','banknote_authentication']
path_to_save = root_path+'/Analysis_results'

# data_i = 'Data5_'

for data_i in data_list:
    print(data_i)
    list_match = [s for s in total_name_list if data_i in s]
    data_total = pd.DataFrame()
    for file in list_match:
        print(list_match)
        os.chdir(root_path + '/Bagging_results')
        name = file[25:]
        data = pd.read_csv(file)
        if ('split4' in name and 'extreme' not in name and 'classic' not in name):
            data.columns = [str(col) + '_split4' for col in data.columns]
        elif ('split2' in name and 'extreme' not in name and 'classic' not in name):
            data.columns = [str(col) + '_split2' for col in data.columns]
        elif ('split9' in name and 'extreme' not in name and 'classic' not in name):
            data.columns = [str(col) + '_split9' for col in data.columns]
        elif ('split4' in name and 'extreme' in name and 'classic' not in name):
            data.columns = [str(col) + '_split4_extreme' for col in data.columns]
        elif ('split2' in name and 'extreme' in name and 'classic' not in name):
            data.columns = [str(col) + '_split2_extreme' for col in data.columns]
        elif ('split9' in name and 'extreme' in name and 'classic' not in name):
            data.columns = [str(col) + '_split9_extreme' for col in data.columns]
        elif ('hard' in name):
            data.columns = [str(col) + '_hard' for col in data.columns]
        elif ('easy' in name):
            data.columns = [str(col) + '_easy' for col in data.columns]
        elif ('combo' in name and 'extreme' not in name and 'split' not in name and 'classic' not in name):
            data.columns = [str(col) + '_combo' for col in data.columns]
        elif ('combo' in name and 'extreme' in name and 'split' not in name and 'classic' not in name):
            data.columns = [str(col) + '_combo_extreme' for col in data.columns]
        elif ('split1' in name and 'extreme' not in name and 'classic' in name):
            data.columns = [str(col) + '_split1_classic' for col in data.columns]
        elif ('split2' in name and 'extreme' not in name and 'classic' in name):
            data.columns = [str(col) + '_split2_classic' for col in data.columns]
        elif ('split4' in name and 'extreme' not in name and 'classic' in name):
            data.columns = [str(col) + '_split4_classic' for col in data.columns]
        elif ('split1' in name and 'extreme' in name and 'classic' in name):
            data.columns = [str(col) + '_split1_classic_extreme' for col in data.columns]
        elif ('split2' in name and 'extreme' in name and 'classic' in name):
            data.columns = [str(col) + '_split2_classic_extreme' for col in data.columns]
        elif ('split4' in name and 'extreme' in name and 'classic' in name):
            data.columns = [str(col) + '_split4_classic_extreme' for col in data.columns]
        data.columns.values[0:2] = ['n_ensemble', 'weights']
        data_total = pd.concat([data_total, data], axis=1)
    data_total = data_total.loc[:, ~data_total.columns.duplicated()]  # remove duplicate columns
    # data_total.columns

    df_long_host = pd.melt(data_total[data_total['weights'] == 'Hostility'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_Hostility_class_mean_easy', 'Boots_Hostility_class_mean_hard',
                                  'Boots_Hostility_class_mean_combo', 'Boots_Hostility_class_mean_combo_extreme',
                                  'Boots_Hostility_class_mean_split2', 'Boots_Hostility_class_mean_split2_extreme',
                                  'Boots_Hostility_class_mean_split4', 'Boots_Hostility_class_mean_split4_extreme',
                                  'Boots_Hostility_class_mean_split9', 'Boots_Hostility_class_mean_split9_extreme',
                                  'Boots_Hostility_class_mean_split1_classic', 'Boots_Hostility_class_mean_split1_classic_extreme',
                                  'Boots_Hostility_class_mean_split2_classic', 'Boots_Hostility_class_mean_split2_classic_extreme',
                                  'Boots_Hostility_class_mean_split4_classic', 'Boots_Hostility_class_mean_split4_classic_extreme'],
                      value_name='Complexity')



    # list of classes
    df_long_host['Complexity'] = df_long_host['Complexity'].str[1:-1] # remove square brackets
    df_class_host = pd.DataFrame(df_long_host['Complexity'].str.split(', ').tolist())  # format
    n_classes = df_class_host.shape[1]
    name = 'Class_'
    col = []
    for t in range(n_classes):
        col.append(name + str(t))
    df_class_host.columns = col
    df_class_host = df_class_host.astype(np.float32)
    df_long_host = pd.concat([df_long_host,df_class_host],axis=1)

    df_long_host.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col

    df_long_host2 = pd.melt(df_long_host,id_vars=['n_ensemble','weights','scheme'],
                      value_vars=col,
                      value_name='Complexity_Class')

    # # Boxplot
    # ax = sns.boxplot(y=df_long_host2["Complexity_Class"], x=df_long_host2["scheme"],hue=df_long_host2["variable"],
    #             order=['Uniform',
    #                    'Boots_Hostility_class_mean_easy', 'Boots_Hostility_class_mean_hard',
    #                               'Boots_Hostility_class_mean_combo', 'Boots_Hostility_class_mean_combo_extreme',
    #                               'Boots_Hostility_class_mean_split2', 'Boots_Hostility_class_mean_split2_extreme',
    #                               'Boots_Hostility_class_mean_split4', 'Boots_Hostility_class_mean_split4_extreme',
    #                               'Boots_Hostility_class_mean_split9', 'Boots_Hostility_class_mean_split9_extreme',
    #                             'Boots_Hostility_class_mean_split1_classic', 'Boots_Hostility_class_mean_split1_classic_extreme',
    #                             'Boots_Hostility_class_mean_split2_classic', 'Boots_Hostility_class_mean_split2_classic_extreme',
    #                             'Boots_Hostility_class_mean_split4_classic', 'Boots_Hostility_class_mean_split4_classic_extreme'
    #             ],
    #                  color='white'
    #                  )
    # sns.stripplot(data=df_long_host2, x="scheme", y="Complexity_Class", hue="variable",
    #               dodge=True, ax=ax,
    #               order=['Uniform',
    #                      'Boots_Hostility_class_mean_easy', 'Boots_Hostility_class_mean_hard',
    #                      'Boots_Hostility_class_mean_combo', 'Boots_Hostility_class_mean_combo_extreme',
    #                      'Boots_Hostility_class_mean_split2', 'Boots_Hostility_class_mean_split2_extreme',
    #                      'Boots_Hostility_class_mean_split4', 'Boots_Hostility_class_mean_split4_extreme',
    #                      'Boots_Hostility_class_mean_split9', 'Boots_Hostility_class_mean_split9_extreme',
    #                   'Boots_Hostility_class_mean_split1_classic', 'Boots_Hostility_class_mean_split1_classic_extreme',
    #                   'Boots_Hostility_class_mean_split2_classic', 'Boots_Hostility_class_mean_split2_classic_extreme',
    #                   'Boots_Hostility_class_mean_split4_classic', 'Boots_Hostility_class_mean_split4_classic_extreme'
    #               ])
    # ax.set_xticklabels(['Uniform',
    #                     'Easy', 'Hard','Combo','Combo_X',
    #                     'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
    #                     'Cl_Split1_3', 'Cl_Split1_3_X', 'Cl_Split2_5', 'Cl_Split2_5_X',
    #                     'Cl_Split4_9', 'Cl_Split4_9_X'],
    #                    rotation=40)
    # plt.legend([], [], frameon=False)
    # plt.show()


    df_long_kdn = pd.melt(data_total[data_total['weights'] == 'kDN'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_kDN_class_mean_easy', 'Boots_kDN_class_mean_hard',
                                'Boots_kDN_class_mean_combo', 'Boots_kDN_class_mean_combo_extreme',
                                'Boots_kDN_class_mean_split2', 'Boots_kDN_class_mean_split2_extreme',
                                'Boots_kDN_class_mean_split4', 'Boots_kDN_class_mean_split4_extreme',
                                'Boots_kDN_class_mean_split9', 'Boots_kDN_class_mean_split9_extreme',
                                'Boots_kDN_class_mean_split1_classic', 'Boots_kDN_class_mean_split1_classic_extreme',
                                'Boots_kDN_class_mean_split2_classic', 'Boots_kDN_class_mean_split2_classic_extreme',
                                'Boots_kDN_class_mean_split4_classic', 'Boots_kDN_class_mean_split4_classic_extreme'],
                      value_name='Complexity')

    df_long_kdn['Complexity'] = df_long_kdn['Complexity'].str[1:-1]  # remove square brackets
    df_class_kdn = pd.DataFrame(df_long_kdn['Complexity'].str.split(', ').tolist())  # format
    df_class_kdn.columns = col
    df_class_kdn = df_class_kdn.astype(np.float32)
    df_long_kdn = pd.concat([df_long_kdn, df_class_kdn], axis=1)
    df_long_kdn.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_kdn2 = pd.melt(df_long_kdn, id_vars=['n_ensemble', 'weights', 'scheme'],
                           value_vars=col,
                           value_name='Complexity_Class')

    df_long_dcp = pd.melt(data_total[data_total['weights'] == 'DCP'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_DCP_class_mean_easy', 'Boots_DCP_class_mean_hard',
                                'Boots_DCP_class_mean_combo', 'Boots_DCP_class_mean_combo_extreme',
                                'Boots_DCP_class_mean_split2', 'Boots_DCP_class_mean_split2_extreme',
                                'Boots_DCP_class_mean_split4', 'Boots_DCP_class_mean_split4_extreme',
                                'Boots_DCP_class_mean_split9', 'Boots_DCP_class_mean_split9_extreme',
                                'Boots_DCP_class_mean_split1_classic', 'Boots_DCP_class_mean_split1_classic_extreme',
                                'Boots_DCP_class_mean_split2_classic', 'Boots_DCP_class_mean_split2_classic_extreme',
                                'Boots_DCP_class_mean_split4_classic', 'Boots_DCP_class_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_dcp['Complexity'] = df_long_dcp['Complexity'].str[1:-1]  # remove square brackets
    df_class_dcp = pd.DataFrame(df_long_dcp['Complexity'].str.split(', ').tolist())  # format
    df_class_dcp.columns = col
    df_class_dcp = df_class_dcp.astype(np.float32)
    df_class_dcp.mean(axis=0)
    df_long_dcp = pd.concat([df_long_dcp, df_class_dcp], axis=1)
    df_long_dcp.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_dcp2 = pd.melt(df_long_dcp, id_vars=['n_ensemble', 'weights', 'scheme'],
                           value_vars=col,
                           value_name='Complexity_Class')


    df_long_lsc = pd.melt(data_total[data_total['weights'] == 'LSC'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_LSC_class_mean_easy', 'Boots_LSC_class_mean_hard',
                            'Boots_LSC_class_mean_combo', 'Boots_LSC_class_mean_combo_extreme',
                            'Boots_LSC_class_mean_split2', 'Boots_LSC_class_mean_split2_extreme',
                            'Boots_LSC_class_mean_split4', 'Boots_LSC_class_mean_split4_extreme',
                            'Boots_LSC_class_mean_split9', 'Boots_LSC_class_mean_split9_extreme',
                            'Boots_LSC_class_mean_split1_classic', 'Boots_LSC_class_mean_split1_classic_extreme',
                            'Boots_LSC_class_mean_split2_classic', 'Boots_LSC_class_mean_split2_classic_extreme',
                            'Boots_LSC_class_mean_split4_classic', 'Boots_LSC_class_mean_split4_classic_extreme'],
                      value_name='Complexity')
    df_long_lsc['Complexity'] = df_long_lsc['Complexity'].str[1:-1]  # remove square brackets
    df_class_lsc = pd.DataFrame(df_long_lsc['Complexity'].str.split(', ').tolist())  # format
    df_class_lsc.columns = col
    df_class_lsc = df_class_lsc.astype(np.float32)
    df_class_lsc.mean(axis=0)
    df_long_lsc = pd.concat([df_long_lsc, df_class_lsc], axis=1)
    df_long_lsc.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_lsc2 = pd.melt(df_long_lsc, id_vars=['n_ensemble', 'weights', 'scheme'],
                           value_vars=col,
                           value_name='Complexity_Class')


    df_long_n1 = pd.melt(data_total[data_total['weights'] == 'N1'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_N1_class_mean_easy', 'Boots_N1_class_mean_hard',
                            'Boots_N1_class_mean_combo', 'Boots_N1_class_mean_combo_extreme',
                            'Boots_N1_class_mean_split2', 'Boots_N1_class_mean_split2_extreme',
                            'Boots_N1_class_mean_split4', 'Boots_N1_class_mean_split4_extreme',
                            'Boots_N1_class_mean_split9', 'Boots_N1_class_mean_split9_extreme',
                            'Boots_N1_class_mean_split1_classic', 'Boots_N1_class_mean_split1_classic_extreme',
                            'Boots_N1_class_mean_split2_classic', 'Boots_N1_class_mean_split2_classic_extreme',
                            'Boots_N1_class_mean_split4_classic', 'Boots_N1_class_mean_split4_classic_extreme'],
                      value_name='Complexity')

    df_long_n1['Complexity'] = df_long_n1['Complexity'].str[1:-1]  # remove square brackets
    df_class_n1 = pd.DataFrame(df_long_n1['Complexity'].str.split(', ').tolist())  # format
    df_class_n1.columns = col
    df_class_n1 = df_class_n1.astype(np.float32)
    df_class_n1.mean(axis=0)
    df_long_n1 = pd.concat([df_long_n1, df_class_n1], axis=1)
    df_long_n1.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_n12 = pd.melt(df_long_n1, id_vars=['n_ensemble', 'weights', 'scheme'],
                          value_vars=col,
                          value_name='Complexity_Class')

    df_long_n2 = pd.melt(data_total[data_total['weights'] == 'N2'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_N2_class_mean_easy', 'Boots_N2_class_mean_hard',
                                'Boots_N2_class_mean_combo', 'Boots_N2_class_mean_combo_extreme',
                                'Boots_N2_class_mean_split2', 'Boots_N2_class_mean_split2_extreme',
                                'Boots_N2_class_mean_split4', 'Boots_N2_class_mean_split4_extreme',
                                'Boots_N2_class_mean_split9', 'Boots_N2_class_mean_split9_extreme',
                                'Boots_N2_class_mean_split1_classic', 'Boots_N2_class_mean_split1_classic_extreme',
                                'Boots_N2_class_mean_split2_classic', 'Boots_N2_class_mean_split2_classic_extreme',
                                'Boots_N2_class_mean_split4_classic', 'Boots_N2_class_mean_split4_classic_extreme'],
                      value_name='Complexity')

    df_long_n2['Complexity'] = df_long_n2['Complexity'].str[1:-1]  # remove square brackets
    df_class_n2 = pd.DataFrame(df_long_n2['Complexity'].str.split(', ').tolist())  # format
    df_class_n2.columns = col
    df_class_n2 = df_class_n2.astype(np.float32)
    df_class_n2.mean(axis=0)
    df_long_n2 = pd.concat([df_long_n2, df_class_n2], axis=1)
    df_long_n2.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_n22 = pd.melt(df_long_n2, id_vars=['n_ensemble', 'weights', 'scheme'],
                          value_vars=col,
                          value_name='Complexity_Class')


    df_long_f1 = pd.melt(data_total[data_total['weights'] == 'F1'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_F1_class_mean_easy', 'Boots_F1_class_mean_hard',
                                'Boots_F1_class_mean_combo', 'Boots_F1_class_mean_combo_extreme',
                                'Boots_F1_class_mean_split2', 'Boots_F1_class_mean_split2_extreme',
                                'Boots_F1_class_mean_split4', 'Boots_F1_class_mean_split4_extreme',
                                'Boots_F1_class_mean_split9', 'Boots_F1_class_mean_split9_extreme',
                                'Boots_F1_class_mean_split1_classic', 'Boots_F1_class_mean_split1_classic_extreme',
                                'Boots_F1_class_mean_split2_classic', 'Boots_F1_class_mean_split2_classic_extreme',
                                'Boots_F1_class_mean_split4_classic', 'Boots_F1_class_mean_split4_classic_extreme'],
                      value_name='Complexity')

    df_long_f1['Complexity'] = df_long_f1['Complexity'].str[1:-1]  # remove square brackets
    df_class_f1 = pd.DataFrame(df_long_f1['Complexity'].str.split(', ').tolist())  # format
    df_class_f1.columns = col
    df_class_f1 = df_class_f1.astype(np.float32)
    df_class_f1.mean(axis=0)
    df_long_f1 = pd.concat([df_long_f1, df_class_f1], axis=1)
    df_long_f1.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_f12 = pd.melt(df_long_f1, id_vars=['n_ensemble', 'weights', 'scheme'],
                          value_vars=col,
                          value_name='Complexity_Class')

    df_long_tdu = pd.melt(data_total[data_total['weights'] == 'TD_U'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_TD_U_class_mean_easy', 'Boots_TD_U_class_mean_hard',
                                'Boots_TD_U_class_mean_combo', 'Boots_TD_U_class_mean_combo_extreme',
                                'Boots_TD_U_class_mean_split2', 'Boots_TD_U_class_mean_split2_extreme',
                                'Boots_TD_U_class_mean_split4', 'Boots_TD_U_class_mean_split4_extreme',
                                'Boots_TD_U_class_mean_split9', 'Boots_TD_U_class_mean_split9_extreme',
                                'Boots_TD_U_class_mean_split1_classic', 'Boots_TD_U_class_mean_split1_classic_extreme',
                                'Boots_TD_U_class_mean_split2_classic', 'Boots_TD_U_class_mean_split2_classic_extreme',
                                'Boots_TD_U_class_mean_split4_classic', 'Boots_TD_U_class_mean_split4_classic_extreme'],
                      value_name='Complexity')

    df_long_tdu['Complexity'] = df_long_tdu['Complexity'].str[1:-1]  # remove square brackets
    df_class_tdu = pd.DataFrame(df_long_tdu['Complexity'].str.split(', ').tolist())  # format
    df_class_tdu.columns = col
    df_class_tdu = df_class_tdu.astype(np.float32)
    df_class_tdu.mean(axis=0)
    df_long_tdu = pd.concat([df_long_tdu, df_class_tdu], axis=1)
    df_long_tdu.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_tdu2 = pd.melt(df_long_tdu, id_vars=['n_ensemble', 'weights', 'scheme'],
                           value_vars=col,
                           value_name='Complexity_Class')


    df_long_cld = pd.melt(data_total[data_total['weights'] == 'CLD'],id_vars=['n_ensemble','weights'],
                      value_vars=['Boots_CLD_class_mean_easy', 'Boots_CLD_class_mean_hard',
                                'Boots_CLD_class_mean_combo', 'Boots_CLD_class_mean_combo_extreme',
                                'Boots_CLD_class_mean_split2', 'Boots_CLD_class_mean_split2_extreme',
                                'Boots_CLD_class_mean_split4', 'Boots_CLD_class_mean_split4_extreme',
                                'Boots_CLD_class_mean_split9', 'Boots_CLD_class_mean_split9_extreme',
                                'Boots_CLD_class_mean_split1_classic', 'Boots_CLD_class_mean_split1_classic_extreme',
                                'Boots_CLD_class_mean_split2_classic', 'Boots_CLD_class_mean_split2_classic_extreme',
                                'Boots_CLD_class_mean_split4_classic', 'Boots_CLD_class_mean_split4_classic_extreme'],
                      value_name='Complexity')

    df_long_cld['Complexity'] = df_long_cld['Complexity'].str[1:-1]  # remove square brackets
    df_class_cld = pd.DataFrame(df_long_cld['Complexity'].str.split(', ').tolist())  # format
    df_class_cld.columns = col
    df_class_cld = df_class_cld.astype(np.float32)
    df_class_cld.mean(axis=0)
    df_long_cld = pd.concat([df_long_cld, df_class_cld], axis=1)
    df_long_cld.columns = ['n_ensemble', 'weights', 'scheme', 'Complexity'] + col
    df_long_cld2 = pd.melt(df_long_cld, id_vars=['n_ensemble', 'weights', 'scheme'],
                           value_vars=col,
                           value_name='Complexity_Class')

    # Uniform information
    data_aux_host = data_total.loc[data_total['weights'] == 'Uniform',['n_ensemble','weights','Boots_Hostility_class_mean_easy']]
    data_aux_host.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_host.reset_index(inplace=True)
    data_aux_host.drop('index',inplace=True,axis=1)
    data_aux_host['Complexity'] = data_aux_host['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_host = pd.DataFrame(data_aux_host['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_host.columns = col
    df_class_uni_host = df_class_uni_host.astype(np.float32)
    data_aux_host = pd.concat([data_aux_host[['n_ensemble','weights']], df_class_uni_host], axis=1)
    data_aux_host2 = pd.melt(data_aux_host,id_vars=['n_ensemble','weights'],
                      value_vars=col,
                      value_name='Complexity_Class')
    data_aux_host2['scheme'] = 'Uniform'

    data_aux_kdn = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_kDN_class_mean_easy']]
    data_aux_kdn.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_kdn.reset_index(inplace=True)
    data_aux_kdn.drop('index', inplace=True, axis=1)
    data_aux_kdn['Complexity'] = data_aux_kdn['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_kdn = pd.DataFrame(data_aux_kdn['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_kdn.columns = col
    df_class_uni_kdn = df_class_uni_kdn.astype(np.float32)
    data_aux_kdn = pd.concat([data_aux_kdn[['n_ensemble', 'weights']], df_class_uni_kdn], axis=1)
    data_aux_kdn2 = pd.melt(data_aux_kdn, id_vars=['n_ensemble', 'weights'],
                            value_vars=col,
                            value_name='Complexity_Class')
    data_aux_kdn2['scheme'] = 'Uniform'

    data_aux_dcp = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_DCP_class_mean_easy']]
    data_aux_dcp.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_dcp.reset_index(inplace=True)
    data_aux_dcp.drop('index', inplace=True, axis=1)
    data_aux_dcp['Complexity'] = data_aux_dcp['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_dcp = pd.DataFrame(data_aux_dcp['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_dcp.columns = col
    df_class_uni_dcp = df_class_uni_dcp.astype(np.float32)
    data_aux_dcp = pd.concat([data_aux_dcp[['n_ensemble', 'weights']], df_class_uni_dcp], axis=1)
    data_aux_dcp2 = pd.melt(data_aux_dcp, id_vars=['n_ensemble', 'weights'],
                            value_vars=col,
                            value_name='Complexity_Class')
    data_aux_dcp2['scheme'] = 'Uniform'

    data_aux_lsc = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_LSC_class_mean_easy']]
    data_aux_lsc.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_lsc.reset_index(inplace=True)
    data_aux_lsc.drop('index', inplace=True, axis=1)
    data_aux_lsc['Complexity'] = data_aux_lsc['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_lsc = pd.DataFrame(data_aux_lsc['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_lsc.columns = col
    df_class_uni_lsc = df_class_uni_lsc.astype(np.float32)
    data_aux_lsc = pd.concat([data_aux_lsc[['n_ensemble', 'weights']], df_class_uni_lsc], axis=1)
    data_aux_lsc2 = pd.melt(data_aux_lsc, id_vars=['n_ensemble', 'weights'],
                            value_vars=col,
                            value_name='Complexity_Class')
    data_aux_lsc2['scheme'] = 'Uniform'

    data_aux_n1 = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_N1_class_mean_easy']]
    data_aux_n1.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_n1.reset_index(inplace=True)
    data_aux_n1.drop('index', inplace=True, axis=1)
    data_aux_n1['Complexity'] = data_aux_n1['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_n1 = pd.DataFrame(data_aux_n1['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_n1.columns = col
    df_class_uni_n1 = df_class_uni_n1.astype(np.float32)
    data_aux_n1 = pd.concat([data_aux_n1[['n_ensemble', 'weights']], df_class_uni_n1], axis=1)
    data_aux_n12 = pd.melt(data_aux_n1, id_vars=['n_ensemble', 'weights'],
                           value_vars=col,
                           value_name='Complexity_Class')
    data_aux_n12['scheme'] = 'Uniform'

    data_aux_n2 = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_N2_class_mean_easy']]
    data_aux_n2.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_n2.reset_index(inplace=True)
    data_aux_n2.drop('index', inplace=True, axis=1)
    data_aux_n2['Complexity'] = data_aux_n2['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_n2 = pd.DataFrame(data_aux_n2['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_n2.columns = col
    df_class_uni_n2 = df_class_uni_n2.astype(np.float32)
    data_aux_n2 = pd.concat([data_aux_n2[['n_ensemble', 'weights']], df_class_uni_n2], axis=1)
    data_aux_n22 = pd.melt(data_aux_n2, id_vars=['n_ensemble', 'weights'],
                           value_vars=col,
                           value_name='Complexity_Class')
    data_aux_n22['scheme'] = 'Uniform'

    data_aux_f1 = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_F1_class_mean_easy']]
    data_aux_f1.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_f1.reset_index(inplace=True)
    data_aux_f1.drop('index', inplace=True, axis=1)
    data_aux_f1['Complexity'] = data_aux_f1['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_f1 = pd.DataFrame(data_aux_f1['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_f1.columns = col
    df_class_uni_f1 = df_class_uni_f1.astype(np.float32)
    data_aux_f1 = pd.concat([data_aux_f1[['n_ensemble', 'weights']], df_class_uni_f1], axis=1)
    data_aux_f12 = pd.melt(data_aux_f1, id_vars=['n_ensemble', 'weights'],
                           value_vars=col,
                           value_name='Complexity_Class')
    data_aux_f12['scheme'] = 'Uniform'

    data_aux_tdu = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_TD_U_class_mean_easy']]
    data_aux_tdu.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_tdu.reset_index(inplace=True)
    data_aux_tdu.drop('index', inplace=True, axis=1)
    data_aux_tdu['Complexity'] = data_aux_tdu['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_tdu = pd.DataFrame(data_aux_tdu['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_tdu.columns = col
    df_class_uni_tdu = df_class_uni_tdu.astype(np.float32)
    data_aux_tdu = pd.concat([data_aux_tdu[['n_ensemble', 'weights']], df_class_uni_tdu], axis=1)
    data_aux_tdu2 = pd.melt(data_aux_tdu, id_vars=['n_ensemble', 'weights'],
                            value_vars=col,
                            value_name='Complexity_Class')
    data_aux_tdu2['scheme'] = 'Uniform'

    data_aux_cld = data_total.loc[
        data_total['weights'] == 'Uniform', ['n_ensemble', 'weights', 'Boots_CLD_class_mean_easy']]
    data_aux_cld.columns = ['n_ensemble', 'weights', 'Complexity']
    data_aux_cld.reset_index(inplace=True)
    data_aux_cld.drop('index', inplace=True, axis=1)
    data_aux_cld['Complexity'] = data_aux_cld['Complexity'].str[1:-1]  # remove square brackets
    df_class_uni_cld = pd.DataFrame(data_aux_cld['Complexity'].str.split(', ').tolist())  # format
    df_class_uni_cld.columns = col
    df_class_uni_cld = df_class_uni_cld.astype(np.float32)
    data_aux_cld = pd.concat([data_aux_cld[['n_ensemble', 'weights']], df_class_uni_cld], axis=1)
    data_aux_cld2 = pd.melt(data_aux_cld, id_vars=['n_ensemble', 'weights'],
                            value_vars=col,
                            value_name='Complexity_Class')
    data_aux_cld2['scheme'] = 'Uniform'


    df_long_host2 = pd.concat([df_long_host2,data_aux_host2],axis=0)
    df_long_kdn2 = pd.concat([df_long_kdn2, data_aux_kdn2], axis=0)
    df_long_dcp2 = pd.concat([df_long_dcp2, data_aux_dcp2], axis=0)
    df_long_lsc2 = pd.concat([df_long_lsc2, data_aux_lsc2], axis=0)
    df_long_n12 = pd.concat([df_long_n12, data_aux_n12], axis=0)
    df_long_n22 = pd.concat([df_long_n22, data_aux_n22], axis=0)
    df_long_f12 = pd.concat([df_long_f12, data_aux_f12], axis=0)
    df_long_tdu2 = pd.concat([df_long_tdu2, data_aux_tdu2], axis=0)
    df_long_cld2 = pd.concat([df_long_cld2, data_aux_cld2], axis=0)


    # Boxplot
    # ax = sns.boxplot(y=df_long_host["Complexity"], x=df_long_host["variable"],
    #             order=['Uniform',
    #                    'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
    #                               'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
    #                               'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
    #                               'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
    #                               'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme'],
    #                  color='white')
    # sns.stripplot(data=df_long_host, x="variable", y="Complexity", dodge=True, ax=ax,
    #               order=['Uniform',
    #                      'Boots_Hostility_dataset_mean_easy', 'Boots_Hostility_dataset_mean_hard',
    #                      'Boots_Hostility_dataset_mean_combo', 'Boots_Hostility_dataset_mean_combo_extreme',
    #                      'Boots_Hostility_dataset_mean_split2', 'Boots_Hostility_dataset_mean_split2_extreme',
    #                      'Boots_Hostility_dataset_mean_split4', 'Boots_Hostility_dataset_mean_split4_extreme',
    #                      'Boots_Hostility_dataset_mean_split9', 'Boots_Hostility_dataset_mean_split9_extreme'])
    # ax.set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
    #                     'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X'],
    #                    rotation=40)
    # plt.show()

    fig, axes = plt.subplots(9, 1, figsize=(9, 50))
    sns.boxplot(ax=axes[0],y=df_long_host2["Complexity_Class"], x=df_long_host2["scheme"],hue=df_long_host2["variable"],
                order=['Uniform',
                       'Boots_Hostility_class_mean_easy', 'Boots_Hostility_class_mean_hard',
                       'Boots_Hostility_class_mean_combo', 'Boots_Hostility_class_mean_combo_extreme',
                       'Boots_Hostility_class_mean_split2', 'Boots_Hostility_class_mean_split2_extreme',
                       'Boots_Hostility_class_mean_split4', 'Boots_Hostility_class_mean_split4_extreme',
                       'Boots_Hostility_class_mean_split9', 'Boots_Hostility_class_mean_split9_extreme',
                       'Boots_Hostility_class_mean_split1_classic',
                       'Boots_Hostility_class_mean_split1_classic_extreme',
                       'Boots_Hostility_class_mean_split2_classic',
                       'Boots_Hostility_class_mean_split2_classic_extreme',
                       'Boots_Hostility_class_mean_split4_classic',
                       'Boots_Hostility_class_mean_split4_classic_extreme'
                       ],
                color='white')
    sns.stripplot(data=df_long_host2, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[0],
                  order=['Uniform',
                          'Boots_Hostility_class_mean_easy', 'Boots_Hostility_class_mean_hard',
                       'Boots_Hostility_class_mean_combo', 'Boots_Hostility_class_mean_combo_extreme',
                       'Boots_Hostility_class_mean_split2', 'Boots_Hostility_class_mean_split2_extreme',
                       'Boots_Hostility_class_mean_split4', 'Boots_Hostility_class_mean_split4_extreme',
                       'Boots_Hostility_class_mean_split9', 'Boots_Hostility_class_mean_split9_extreme',
                       'Boots_Hostility_class_mean_split1_classic',
                       'Boots_Hostility_class_mean_split1_classic_extreme',
                       'Boots_Hostility_class_mean_split2_classic',
                       'Boots_Hostility_class_mean_split2_classic_extreme',
                       'Boots_Hostility_class_mean_split4_classic',
                       'Boots_Hostility_class_mean_split4_classic_extreme'
                         ])
    axes[0].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                       rotation=40)
    axes[0].legend([], [], frameon=False)
    axes[0].set(ylabel='Hostility')
    sns.boxplot(ax=axes[1],y=df_long_kdn2["Complexity_Class"], x=df_long_kdn2["scheme"],hue=df_long_kdn2["variable"],
                order=['Uniform',
                       'Boots_kDN_class_mean_easy', 'Boots_kDN_class_mean_hard',
                       'Boots_kDN_class_mean_combo', 'Boots_kDN_class_mean_combo_extreme',
                       'Boots_kDN_class_mean_split2', 'Boots_kDN_class_mean_split2_extreme',
                       'Boots_kDN_class_mean_split4', 'Boots_kDN_class_mean_split4_extreme',
                       'Boots_kDN_class_mean_split9', 'Boots_kDN_class_mean_split9_extreme',
                       'Boots_kDN_class_mean_split1_classic', 'Boots_kDN_class_mean_split1_classic_extreme',
                       'Boots_kDN_class_mean_split2_classic', 'Boots_kDN_class_mean_split2_classic_extreme',
                       'Boots_kDN_class_mean_split4_classic', 'Boots_kDN_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_kdn2, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[1],
                  order=['Uniform',
                         'Boots_kDN_class_mean_easy', 'Boots_kDN_class_mean_hard',
                         'Boots_kDN_class_mean_combo', 'Boots_kDN_class_mean_combo_extreme',
                         'Boots_kDN_class_mean_split2', 'Boots_kDN_class_mean_split2_extreme',
                         'Boots_kDN_class_mean_split4', 'Boots_kDN_class_mean_split4_extreme',
                         'Boots_kDN_class_mean_split9', 'Boots_kDN_class_mean_split9_extreme',
                         'Boots_kDN_class_mean_split1_classic', 'Boots_kDN_class_mean_split1_classic_extreme',
                         'Boots_kDN_class_mean_split2_classic', 'Boots_kDN_class_mean_split2_classic_extreme',
                         'Boots_kDN_class_mean_split4_classic', 'Boots_kDN_class_mean_split4_classic_extreme'])
    axes[1].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                       rotation=40)
    axes[1].legend([], [], frameon=False)
    axes[1].set(ylabel='kDN')
    sns.boxplot(ax=axes[2], y=df_long_dcp2["Complexity_Class"], x=df_long_dcp2["scheme"],hue=df_long_dcp2["variable"],
                order=['Uniform',
                       'Boots_DCP_class_mean_easy', 'Boots_DCP_class_mean_hard',
                       'Boots_DCP_class_mean_combo', 'Boots_DCP_class_mean_combo_extreme',
                       'Boots_DCP_class_mean_split2', 'Boots_DCP_class_mean_split2_extreme',
                       'Boots_DCP_class_mean_split4', 'Boots_DCP_class_mean_split4_extreme',
                       'Boots_DCP_class_mean_split9', 'Boots_DCP_class_mean_split9_extreme',
                       'Boots_DCP_class_mean_split1_classic', 'Boots_DCP_class_mean_split1_classic_extreme',
                       'Boots_DCP_class_mean_split2_classic', 'Boots_DCP_class_mean_split2_classic_extreme',
                       'Boots_DCP_class_mean_split4_classic', 'Boots_DCP_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_dcp2, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[2],
                  order=['Uniform',
                         'Boots_DCP_class_mean_easy', 'Boots_DCP_class_mean_hard',
                         'Boots_DCP_class_mean_combo', 'Boots_DCP_class_mean_combo_extreme',
                         'Boots_DCP_class_mean_split2', 'Boots_DCP_class_mean_split2_extreme',
                         'Boots_DCP_class_mean_split4', 'Boots_DCP_class_mean_split4_extreme',
                         'Boots_DCP_class_mean_split9', 'Boots_DCP_class_mean_split9_extreme',
                         'Boots_DCP_class_mean_split1_classic', 'Boots_DCP_class_mean_split1_classic_extreme',
                         'Boots_DCP_class_mean_split2_classic', 'Boots_DCP_class_mean_split2_classic_extreme',
                         'Boots_DCP_class_mean_split4_classic', 'Boots_DCP_class_mean_split4_classic_extreme'])
    axes[2].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[2].legend([], [], frameon=False)
    axes[2].set(ylabel='DCP')
    sns.boxplot(ax=axes[3], y=df_long_lsc2["Complexity_Class"], x=df_long_lsc2["scheme"],hue=df_long_lsc2["variable"],
                order=['Uniform',
                       'Boots_LSC_class_mean_easy', 'Boots_LSC_class_mean_hard',
                       'Boots_LSC_class_mean_combo', 'Boots_LSC_class_mean_combo_extreme',
                       'Boots_LSC_class_mean_split2', 'Boots_LSC_class_mean_split2_extreme',
                       'Boots_LSC_class_mean_split4', 'Boots_LSC_class_mean_split4_extreme',
                       'Boots_LSC_class_mean_split9', 'Boots_LSC_class_mean_split9_extreme',
                       'Boots_LSC_class_mean_split1_classic', 'Boots_LSC_class_mean_split1_classic_extreme',
                       'Boots_LSC_class_mean_split2_classic', 'Boots_LSC_class_mean_split2_classic_extreme',
                       'Boots_LSC_class_mean_split4_classic', 'Boots_LSC_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_lsc2, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[3],
                  order=['Uniform',
                         'Boots_LSC_class_mean_easy', 'Boots_LSC_class_mean_hard',
                         'Boots_LSC_class_mean_combo', 'Boots_LSC_class_mean_combo_extreme',
                         'Boots_LSC_class_mean_split2', 'Boots_LSC_class_mean_split2_extreme',
                         'Boots_LSC_class_mean_split4', 'Boots_LSC_class_mean_split4_extreme',
                         'Boots_LSC_class_mean_split9', 'Boots_LSC_class_mean_split9_extreme',
                         'Boots_LSC_class_mean_split1_classic', 'Boots_LSC_class_mean_split1_classic_extreme',
                         'Boots_LSC_class_mean_split2_classic', 'Boots_LSC_class_mean_split2_classic_extreme',
                         'Boots_LSC_class_mean_split4_classic', 'Boots_LSC_class_mean_split4_classic_extreme'])
    axes[3].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[3].legend([], [], frameon=False)
    axes[3].set(ylabel='LSC')
    sns.boxplot(ax=axes[4], y=df_long_n12["Complexity_Class"], x=df_long_n12["scheme"],hue=df_long_n12["variable"],
                order=['Uniform',
                       'Boots_N1_class_mean_easy', 'Boots_N1_class_mean_hard',
                       'Boots_N1_class_mean_combo', 'Boots_N1_class_mean_combo_extreme',
                       'Boots_N1_class_mean_split2', 'Boots_N1_class_mean_split2_extreme',
                       'Boots_N1_class_mean_split4', 'Boots_N1_class_mean_split4_extreme',
                       'Boots_N1_class_mean_split9', 'Boots_N1_class_mean_split9_extreme',
                       'Boots_N1_class_mean_split1_classic', 'Boots_N1_class_mean_split1_classic_extreme',
                       'Boots_N1_class_mean_split2_classic', 'Boots_N1_class_mean_split2_classic_extreme',
                       'Boots_N1_class_mean_split4_classic', 'Boots_N1_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_n12, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[4],
                  order=['Uniform',
                         'Boots_N1_class_mean_easy', 'Boots_N1_class_mean_hard',
                         'Boots_N1_class_mean_combo', 'Boots_N1_class_mean_combo_extreme',
                         'Boots_N1_class_mean_split2', 'Boots_N1_class_mean_split2_extreme',
                         'Boots_N1_class_mean_split4', 'Boots_N1_class_mean_split4_extreme',
                         'Boots_N1_class_mean_split9', 'Boots_N1_class_mean_split9_extreme',
                         'Boots_N1_class_mean_split1_classic', 'Boots_N1_class_mean_split1_classic_extreme',
                         'Boots_N1_class_mean_split2_classic', 'Boots_N1_class_mean_split2_classic_extreme',
                         'Boots_N1_class_mean_split4_classic', 'Boots_N1_class_mean_split4_classic_extreme'])
    axes[4].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[4].legend([], [], frameon=False)
    axes[4].set(ylabel='N1')
    sns.boxplot(ax=axes[5], y=df_long_n22["Complexity_Class"], x=df_long_n22["scheme"],hue=df_long_n22["variable"],
                order=['Uniform',
                       'Boots_N2_class_mean_easy', 'Boots_N2_class_mean_hard',
                       'Boots_N2_class_mean_combo', 'Boots_N2_class_mean_combo_extreme',
                       'Boots_N2_class_mean_split2', 'Boots_N2_class_mean_split2_extreme',
                       'Boots_N2_class_mean_split4', 'Boots_N2_class_mean_split4_extreme',
                       'Boots_N2_class_mean_split9', 'Boots_N2_class_mean_split9_extreme',
                       'Boots_N2_class_mean_split1_classic', 'Boots_N2_class_mean_split1_classic_extreme',
                       'Boots_N2_class_mean_split2_classic', 'Boots_N2_class_mean_split2_classic_extreme',
                       'Boots_N2_class_mean_split4_classic', 'Boots_N2_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_n22, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[5],
                  order=['Uniform',
                         'Boots_N2_class_mean_easy', 'Boots_N2_class_mean_hard',
                         'Boots_N2_class_mean_combo', 'Boots_N2_class_mean_combo_extreme',
                         'Boots_N2_class_mean_split2', 'Boots_N2_class_mean_split2_extreme',
                         'Boots_N2_class_mean_split4', 'Boots_N2_class_mean_split4_extreme',
                         'Boots_N2_class_mean_split9', 'Boots_N2_class_mean_split9_extreme',
                         'Boots_N2_class_mean_split1_classic', 'Boots_N2_class_mean_split1_classic_extreme',
                         'Boots_N2_class_mean_split2_classic', 'Boots_N2_class_mean_split2_classic_extreme',
                         'Boots_N2_class_mean_split4_classic', 'Boots_N2_class_mean_split4_classic_extreme'])
    axes[5].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[5].legend([], [], frameon=False)
    axes[5].set(ylabel='N2')
    sns.boxplot(ax=axes[6], y=df_long_f12["Complexity_Class"], x=df_long_f12["scheme"],hue=df_long_f12["variable"],
                order=['Uniform',
                       'Boots_F1_class_mean_easy', 'Boots_F1_class_mean_hard',
                       'Boots_F1_class_mean_combo', 'Boots_F1_class_mean_combo_extreme',
                       'Boots_F1_class_mean_split2', 'Boots_F1_class_mean_split2_extreme',
                       'Boots_F1_class_mean_split4', 'Boots_F1_class_mean_split4_extreme',
                       'Boots_F1_class_mean_split9', 'Boots_F1_class_mean_split9_extreme',
                       'Boots_F1_class_mean_split1_classic', 'Boots_F1_class_mean_split1_classic_extreme',
                       'Boots_F1_class_mean_split2_classic', 'Boots_F1_class_mean_split2_classic_extreme',
                       'Boots_F1_class_mean_split4_classic', 'Boots_F1_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_f12, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[6],
                  order=['Uniform',
                         'Boots_F1_class_mean_easy', 'Boots_F1_class_mean_hard',
                         'Boots_F1_class_mean_combo', 'Boots_F1_class_mean_combo_extreme',
                         'Boots_F1_class_mean_split2', 'Boots_F1_class_mean_split2_extreme',
                         'Boots_F1_class_mean_split4', 'Boots_F1_class_mean_split4_extreme',
                         'Boots_F1_class_mean_split9', 'Boots_F1_class_mean_split9_extreme',
                         'Boots_F1_class_mean_split1_classic', 'Boots_F1_class_mean_split1_classic_extreme',
                         'Boots_F1_class_mean_split2_classic', 'Boots_F1_class_mean_split2_classic_extreme',
                         'Boots_F1_class_mean_split4_classic', 'Boots_F1_class_mean_split4_classic_extreme'])
    axes[6].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[6].legend([], [], frameon=False)
    axes[6].set(ylabel='F1')
    sns.boxplot(ax=axes[7], y=df_long_tdu2["Complexity_Class"], x=df_long_tdu2["scheme"],hue=df_long_tdu2["variable"],
                order=['Uniform',
                       'Boots_TD_U_class_mean_easy', 'Boots_TD_U_class_mean_hard',
                       'Boots_TD_U_class_mean_combo', 'Boots_TD_U_class_mean_combo_extreme',
                       'Boots_TD_U_class_mean_split2', 'Boots_TD_U_class_mean_split2_extreme',
                       'Boots_TD_U_class_mean_split4', 'Boots_TD_U_class_mean_split4_extreme',
                       'Boots_TD_U_class_mean_split9', 'Boots_TD_U_class_mean_split9_extreme',
                       'Boots_TD_U_class_mean_split1_classic', 'Boots_TD_U_class_mean_split1_classic_extreme',
                       'Boots_TD_U_class_mean_split2_classic', 'Boots_TD_U_class_mean_split2_classic_extreme',
                       'Boots_TD_U_class_mean_split4_classic', 'Boots_TD_U_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_tdu2, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[7],
                  order=['Uniform',
                         'Boots_TD_U_class_mean_easy', 'Boots_TD_U_class_mean_hard',
                            'Boots_TD_U_class_mean_combo', 'Boots_TD_U_class_mean_combo_extreme',
                            'Boots_TD_U_class_mean_split2', 'Boots_TD_U_class_mean_split2_extreme',
                            'Boots_TD_U_class_mean_split4', 'Boots_TD_U_class_mean_split4_extreme',
                            'Boots_TD_U_class_mean_split9', 'Boots_TD_U_class_mean_split9_extreme',
                            'Boots_TD_U_class_mean_split1_classic', 'Boots_TD_U_class_mean_split1_classic_extreme',
                            'Boots_TD_U_class_mean_split2_classic', 'Boots_TD_U_class_mean_split2_classic_extreme',
                            'Boots_TD_U_class_mean_split4_classic', 'Boots_TD_U_class_mean_split4_classic_extreme'])
    axes[7].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[7].legend([], [], frameon=False)
    axes[7].set(ylabel='TDU')
    sns.boxplot(ax=axes[8], y=df_long_cld2["Complexity_Class"], x=df_long_cld2["scheme"],hue=df_long_cld2["variable"],
                order=['Uniform',
                        'Boots_CLD_class_mean_easy', 'Boots_CLD_class_mean_hard',
                            'Boots_CLD_class_mean_combo', 'Boots_CLD_class_mean_combo_extreme',
                            'Boots_CLD_class_mean_split2', 'Boots_CLD_class_mean_split2_extreme',
                            'Boots_CLD_class_mean_split4', 'Boots_CLD_class_mean_split4_extreme',
                            'Boots_CLD_class_mean_split9', 'Boots_CLD_class_mean_split9_extreme',
                            'Boots_CLD_class_mean_split1_classic', 'Boots_CLD_class_mean_split1_classic_extreme',
                            'Boots_CLD_class_mean_split2_classic', 'Boots_CLD_class_mean_split2_classic_extreme',
                            'Boots_CLD_class_mean_split4_classic', 'Boots_CLD_class_mean_split4_classic_extreme'],
                color='white')
    sns.stripplot(data=df_long_cld2, x="scheme", y="Complexity_Class", hue="variable",
                  dodge=True, ax=axes[8],
                  order=['Uniform',
                         'Boots_CLD_class_mean_easy', 'Boots_CLD_class_mean_hard',
                            'Boots_CLD_class_mean_combo', 'Boots_CLD_class_mean_combo_extreme',
                            'Boots_CLD_class_mean_split2', 'Boots_CLD_class_mean_split2_extreme',
                            'Boots_CLD_class_mean_split4', 'Boots_CLD_class_mean_split4_extreme',
                            'Boots_CLD_class_mean_split9', 'Boots_CLD_class_mean_split9_extreme',
                            'Boots_CLD_class_mean_split1_classic', 'Boots_CLD_class_mean_split1_classic_extreme',
                            'Boots_CLD_class_mean_split2_classic', 'Boots_CLD_class_mean_split2_classic_extreme',
                            'Boots_CLD_class_mean_split4_classic', 'Boots_CLD_class_mean_split4_classic_extreme'])
    axes[8].set_xticklabels(['Uniform', 'Easy', 'Hard','Combo','Combo_X',
                        'Split2','Split2_X','Split4','Split4_X','Split9','Split9_X',
                        'Cl_Split1_3','Cl_Split1_3_X','Cl_Split2_5','Cl_Split2_5_X',
                             'Cl_Split4_9','Cl_Split4_9_X'],
                            rotation=40)
    axes[8].legend([], [], frameon=False)
    axes[8].set(ylabel='CLD')

    # plt.show()
    name_plot = 'ComplexityAnalysisClass_' + str(data_i) + '.png'
    os.chdir(path_to_save)
    plt.savefig(name_plot)
    plt.clf()








