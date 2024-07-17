####################### BAGGING CODE WITH COMPLEXITY-DRIVEN SAMPLING: GENERAL ALGORITHM ######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from aux_functions import aggregation_results_final_algorith
import multiprocessing as mp

root_path = os.getcwd()



def bootstrap_sample(X_train, y_train, weights):
    n_train = len(y_train)
    # Indices corresponding to a weighted sampling with replacement of the same sample
    # size than the original data
    np.random.seed(1)
    bootstrap_indices = random.choices(np.arange(y_train.shape[0]), weights=weights, k=n_train)

    X_bootstrap = X_train[bootstrap_indices]
    y_bootstrap = y_train[bootstrap_indices]

    return X_bootstrap, y_bootstrap, bootstrap_indices

def voting_rule(preds):

    mode_preds = preds.mode(axis=1)  # most common pred value
    if (mode_preds.shape[1] > 1):
        mode_preds_aux = mode_preds.dropna()  # cases with more than one most common value (= ties)
        np.random.seed(1)
        mode_preds_aux = mode_preds_aux.apply(random.choice, axis=1)  # ties are broken randomly

        mode_preds.iloc[mode_preds_aux.index, 0] = mode_preds_aux

    # Once the ties problem is solved, first column contains the final ensemble predictions
    preds_final = mode_preds[0]

    return preds_final





def StandardBagging(data,n_ensembles,n_ensembles_v, name_data,path_to_save, stump):

    # X, y
    X = data.iloc[:,:-1].to_numpy() # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy()

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy',
                                    'Boots_Hostility_dataset','Boots_kDN_dataset','Boots_DCP_dataset',
                                    'Boots_TD_U_dataset','Boots_CLD_dataset', 'Boots_N1_dataset',
                                    'Boots_N2_dataset','Boots_LSC_dataset','Boots_F1_dataset',
                                    'Boots_Hostility_class', 'Boots_kDN_class','Boots_DCP_class',
                                    'Boots_TD_U_class','Boots_CLD_class','Boots_N1_class',
                                    'Boots_N2_class','Boots_LSC_class','Boots_F1_class'])

    skf = StratifiedKFold(n_splits=10, random_state=1,shuffle=True)
    fold = 0
    for train_index, test_index in skf.split(X, y):
        fold = fold + 1
        print(fold)
        # print(train_index)
        # print(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_test)
        # print(y_test)

        # Obtain complexity measures on train set
        data_train = pd.DataFrame(X_train)
        data_train['y'] = y_train
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train,False,None, None)
        # Selection of complexity measures
        df_measures_sel = df_measures[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1','y']]

        weights = np.repeat(1 / len(y_train), len(y_train), axis=0)

        preds = pd.DataFrame()
        ensemble_preds = pd.DataFrame()

        for i in range(n_ensembles):

            np.random.seed(1)
            X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)

            # Save complexity information (class and dataset levels)
            df_measures_bootstrap = df_measures_sel.iloc[bootstrap_indices].copy()
            # Binarize hostility to obtain class and dataset levels
            df_measures_bootstrap.loc[:, 'Hostility_bin'] = np.where(df_measures_bootstrap['Hostility'] >= 0.5, 1, 0)

            # Dataset
            [Boots_kDN_dataset, Boots_DCP_dataset, Boots_TD_U_dataset,
             Boots_CLD_dataset, Boots_N1_dataset, Boots_N2_dataset,
             Boots_LSC_dataset, Boots_F1_dataset] = df_measures_bootstrap.mean()[1:-2]
            Boots_Hostility_dataset = df_measures_bootstrap.mean()['Hostility_bin']
            # Class
            df_classes_boots = df_measures_bootstrap.groupby('y').mean()
            Boots_Hostility_class = df_classes_boots['Hostility_bin'].tolist()
            Boots_kDN_class = df_classes_boots['kDN'].tolist()
            Boots_DCP_class = df_classes_boots['DCP'].tolist()
            Boots_TD_U_class = df_classes_boots['TD_U'].tolist()
            Boots_CLD_class = df_classes_boots['CLD'].tolist()
            Boots_LSC_class = df_classes_boots['LSC'].tolist()
            Boots_N1_class = df_classes_boots['N1'].tolist()
            Boots_N2_class = df_classes_boots['N2'].tolist()
            Boots_F1_class = df_classes_boots['F1'].tolist()

            # Train DT in bootstrap sample and test y X_test, y_test
            if (stump == 'no'):
                clf = DecisionTreeClassifier(random_state=0)
            else:  # Decision Stump
                clf = DecisionTreeClassifier(max_depth=1, random_state=0)
            clf.fit(X_bootstrap, y_bootstrap)
            y_pred = clf.predict(X_test)

            if (i == 0):  # first iteration
                col_name = 'pred_' + str(i)
                preds[col_name] = y_pred  # individual predictions
                # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
                y_predicted = y_pred
            else:
                col_name = 'pred_' + str(i)
                preds[col_name] = y_pred  # individual predictions
                votes = voting_rule(preds)
                # print(votes)
                votes_dict = {'col_name': votes}
                votes_df = pd.DataFrame(votes_dict)
                votes_df.columns = [col_name]
                # ensemble_preds[col_name] = votes # ensemble prediction with majority voting rule
                ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)
                # print(ensemble_preds[col_name])

                y_predicted = ensemble_preds.iloc[:, -1:]  # last column
            acc = accuracy_score(y_predicted, y_test)
            conf_matrix = confusion_matrix(y_test, y_predicted).tolist()

            if (i in n_ensembles_v):

                results_dict = {'dataset': name_data, 'fold': fold, 'n_ensemble': i, 'weights': 'Uniform',
                                'confusion_matrix': [conf_matrix], 'accuracy': acc,
                                'Boots_Hostility_dataset': Boots_Hostility_dataset,
                                'Boots_kDN_dataset': Boots_kDN_dataset,
                                'Boots_DCP_dataset': Boots_DCP_dataset,
                                'Boots_TD_U_dataset': Boots_TD_U_dataset,
                                'Boots_CLD_dataset': Boots_CLD_dataset,
                                'Boots_N1_dataset': Boots_N1_dataset,
                                'Boots_N2_dataset': Boots_N2_dataset,
                                'Boots_LSC_dataset': Boots_LSC_dataset,
                                'Boots_F1_dataset': Boots_F1_dataset,

                                'Boots_Hostility_class': [Boots_Hostility_class],
                                'Boots_kDN_class': [Boots_kDN_class],
                                'Boots_DCP_class': [Boots_DCP_class],
                                'Boots_TD_U_class': [Boots_TD_U_class],
                                'Boots_CLD_class': [Boots_CLD_class],
                                'Boots_N1_class': [Boots_N1_class],
                                'Boots_N2_class': [Boots_N2_class],
                                'Boots_LSC_class': [Boots_LSC_class],
                                'Boots_F1_class': [Boots_F1_class]}
                results_aux = pd.DataFrame(results_dict, index=[0])
                results = pd.concat([results, results_aux])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'StandardBagging_' + name_data + '.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)

    ##### Agregation of results
    df_aggre = aggregation_results_final_algorith(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_StandardBagging_' + name_data + '.csv'
    df_aggre.to_csv(nombre_csv_agg, encoding='utf_8_sig',index=False)

    return results, df_aggre





#
# path_csv = os.chdir(root_path+'/datasets')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv'):
#         total_name_list.append(filename)
#
#
# path_to_save = root_path+'/Results_StandardBagging'
# n_ensembles = 200 # maximum number of ensembles to consider (later we plot and stop when we want)
# # CM_selected = 'Hostility' # selection of the complexity measure to guide the sampling
# n_ensembles_v = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
#                  109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
#
# for data_file in total_name_list:
#     os.chdir(root_path + '/datasets')
#     print(data_file)
#     file = data_file
#     name_data = data_file[:-4]
#     data = pd.read_csv(file)
#     stump = 'no'
#     results, df_aggre = StandardBagging(data,n_ensembles, name_data,path_to_save, stump)


####################################
#####   PARALLELIZED VERSION   #####
####################################

def results_StandardBagging(data_file):
    n_ensembles = 300  # maximum number of ensembles to consider (later we plot and stop when we want)
    # n_ensembles_v = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
    #                  109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
    n_ensembles_v = list(np.arange(0,301,1)) # los saco todos porque son acumulativos
    path_to_save = root_path + '/Results_StandardBagging'

    # for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    name_data = data_file[:-4]
    data = pd.read_csv(data_file)
    stump = 'no'
    StandardBagging(data,n_ensembles,n_ensembles_v, name_data,path_to_save, stump)

    return

# total_name_list

path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)

N= mp.cpu_count()

with mp.Pool(processes = 2) as p:
        p.map(results_StandardBagging, [data_file for data_file in total_name_list])
        # p.close()




###########################################################################################
#####    Leemos todos los resultados y seleccionamos el mejor ensemble para cada caso #####
###########################################################################################

# path_csv = os.chdir(root_path+'/Results_StandardBagging')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv') and 'Aggregated' in filename:
#         total_name_list.append(filename)
#
# best_n_trees_df = pd.DataFrame()
#
# for file_i in total_name_list:
#     name_data = file_i[34:-4]
#     data_file = pd.read_csv(file_i)
#     index_max_acc = data_file.accuracy_mean.argmax()
#     max_acc = data_file.accuracy_mean.max()
#     best_n_ensemble = data_file.iloc[index_max_acc, 0]
#     best_n_ensemble_df = pd.DataFrame({'name_data':name_data,
#                                        'best_n_ensemble':best_n_ensemble,
#                                        'best_acc':max_acc}, index=[0])
#     best_n_trees_df = pd.concat([best_n_trees_df,best_n_ensemble_df])
#

