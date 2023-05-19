################### SCRIPT TO ANALYZE COMPLEXITY OF THE BOOTSTRAP SAMPLES WITH EVERY STRATEGY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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



data_list = ['Data1_','Data2','Data3','Data4','Data5','Data6','Data7','Data8',
             'Data9','Data10','Data11','Data12','Data13',
             'pima','segment','ilpd','diabetes','ionosphere','sonar','wdbc']
path_to_save = root_path+'/Analysis_results'

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










