###################------------- CODIGO USADO EN EL IDEAL -----------------############################
####################### BAGGING CODE WITH COMPLEXITY-DRIVEN SAMPLING ######################################
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
from aux_functions import aggregation_results

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



def complexity_driven_bagging(X,y,n_ensembles, name_data,path_to_save, emphasis):

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy',
                                    'Boots_Hostility_dataset','Boots_kDN_dataset','Boots_DCP_dataset',
                                    'Boots_TD_U_dataset','Boots_CLD_dataset', 'Boots_N1_dataset',
                                    'Boots_N2_dataset','Boots_LSC_dataset','Boots_F1_dataset',
                                    'Boots_Hostility_class', 'Boots_kDN_class','Boots_DCP_class',
                                    'Boots_TD_U_class','Boots_CLD_class','Boots_N1_class',
                                    'Boots_N2_class','Boots_LSC_class','Boots_F1_class'])

    # Complexity measures list to check
    # CM_list = ['L1']
    CM_list = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1', 'Uniform']
    # CM_selected = 'Hostility'

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

        for CM_selected in CM_list:
            print(CM_selected)

            if (CM_selected == 'Uniform'): # classic Bagging with uniform probability sampling
                weights = np.repeat(1/len(y_train), len(y_train), axis=0)
            else: # Sampling using Complexity measures
                CM_values = df_measures[CM_selected]
                if (emphasis == 'easy'):
                    ranking = CM_values.rank(method='average', ascending=False)  # more weight to easy
                elif (emphasis == 'hard'):
                    ranking = CM_values.rank(method='average', ascending=True) # more weight to difficult
                # elif (emphasis == '1n_hard'):
                #     ranking = np.repeat(1/len(y_train), len(y_train), axis=0) + CM_values # more weight to difficult 1/n +
                # elif (emphasis == '1n_easy'):
                #     ranking = np.repeat(1/len(y_train), len(y_train), axis=0) + (1 - CM_values) # more weight to easy 1/n +
                # elif (emphasis == 'classes_1n_hard'):
                #     ## If we make per class specifically
                #     y_train_aux = np.concatenate(y_train, axis=0)
                #     n_classes = len(np.unique(y_train))
                #     ranking_aux = np.zeros(len(y_train_aux))
                #     ranking = np.zeros(len(y_train_aux))
                #     for c in range(n_classes):
                #         # print(c)
                #         n_class_c = np.sum(y_train_aux == c)
                #         ranking_aux[y_train_aux == c] = (1 / n_class_c) + CM_values[y_train_aux == c]
                #         ranking[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                # elif (emphasis == 'classes_1n_easy'):
                #     ## If we make per class specifically
                #     y_train_aux = np.concatenate(y_train, axis=0)
                #     n_classes = len(np.unique(y_train))
                #     ranking_aux = np.zeros(len(y_train_aux))
                #     ranking = np.zeros(len(y_train_aux))
                #     for c in range(n_classes):
                #         # print(c)
                #         n_class_c = np.sum(y_train_aux == c)
                #         ranking_aux[y_train_aux == c] = (1 / n_class_c) + (1 - CM_values[y_train_aux == c])
                #         ranking[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                elif (emphasis == 'classes_hard'):
                    ## If we make per class specifically: ranking
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=True)
                        ranking[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                elif (emphasis == 'classes_easy'):
                    ## If we make per class specifically: ranking
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to easy
                        ranking_aux[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=False)
                        ranking[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                # elif (emphasis == 'frontier'):
                #     n_classes = len(np.unique(y_train))
                #     max_uncertainty = 1 / n_classes
                #     min_value = min(max_uncertainty-0.05,0.35)
                #     CM_values_aux = np.zeros(len(CM_values))
                #     CM_values_aux[(CM_values<0.65) & (CM_values>min_value)] = 1 # borderline points
                #     CM_values_aux[(CM_values <= min_value)] = 2 # easy points
                #     CM_values_aux[(CM_values >= 0.65)] = 3  # difficult points
                #     # aa = pd.DataFrame(CM_values_aux).rank(method='average', ascending=False)
                #     # np.unique(aa)/(128+2015+2400)
                #     # We select the thresholds manually
                #     CM_values_aux[CM_values_aux == 1] = 0.5 / sum(CM_values_aux == 1)
                #     CM_values_aux[CM_values_aux == 2] = 0.35 / sum(CM_values_aux == 2)
                #     CM_values_aux[CM_values_aux == 3] = 0.15 / sum(CM_values_aux == 3)
                #     ranking = CM_values_aux

                weights = ranking/sum(ranking) # probability distribution

            preds = pd.DataFrame()
            ensemble_preds = pd.DataFrame()
            # i = 0
            for i in range(n_ensembles):

                # print(i)
                # Get bootstrap sample following CM_weights
                n_train = len(y_train)
                np.random.seed(1)

                X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)

                # Save complexity information (class and dataset levels)
                # with this code the measures are recalculated
                # data_bootstrap = pd.DataFrame(X_bootstrap, columns=['x1', 'x2'])
                # data_bootstrap['y'] = y_bootstrap
                # _, df_classes_dataset_bootstrap = all_measures(data_bootstrap, False, None, None)
                # info_complexity_dataset = df_classes_dataset_bootstrap[CM_selected]['dataset']
                # info_complexity_class = df_classes_dataset_bootstrap[CM_selected][:-1].tolist()

                # Save complexity information (class and dataset levels)
                df_measures_bootstrap = df_measures_sel.iloc[bootstrap_indices].copy()
                # Binarize hostility to obtain class and dataset levels
                df_measures_bootstrap.loc[:, 'Hostility_bin'] = np.where(df_measures_bootstrap['Hostility'] >= 0.5, 1,0)

                # Dataset
                [Boots_kDN_dataset,Boots_DCP_dataset,Boots_TD_U_dataset,
                 Boots_CLD_dataset,Boots_N1_dataset,Boots_N2_dataset,
                 Boots_LSC_dataset,Boots_F1_dataset] = df_measures_bootstrap.mean()[1:-2]
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
                clf = DecisionTreeClassifier(random_state=0)
                clf.fit(X_bootstrap, y_bootstrap)
                y_pred = clf.predict(X_test)

                if (i==0): # first iteration
                    col_name = 'pred_' + str(i)
                    preds[col_name] = y_pred  # individual predictions
                    # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
                    y_predicted = y_pred
                else:
                    col_name = 'pred_'+str(i)
                    preds[col_name] = y_pred # individual predictions
                    votes = voting_rule(preds)
                    # print(votes)
                    votes_dict = {'col_name':votes}
                    votes_df = pd.DataFrame(votes_dict)
                    votes_df.columns = [col_name]
                    # ensemble_preds[col_name] = votes # ensemble prediction with majority voting rule
                    ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)
                    # print(ensemble_preds[col_name])

                    y_predicted = ensemble_preds.iloc[:, -1:] # last column
                acc = accuracy_score(y_predicted, y_test)
                conf_matrix = confusion_matrix(y_test, y_predicted).tolist()

                results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                                    'confusion_matrix':[conf_matrix], 'accuracy':acc,
                                    'Boots_Hostility_dataset':Boots_Hostility_dataset,
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
                results = pd.concat([results,results_aux])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Bagging_' + name_data + '_MoreWeight_' + emphasis +'_Instances.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)


    ##### Agregation of results
    # cols_numeric = results.select_dtypes([np.number]).columns
    # df_aggre_mean = results.groupby(['n_ensemble', 'weights'], as_index=False)[cols_numeric].mean()
    # df_aggre_mean.columns = ['n_ensemble', 'weights', 'accuracy_mean',
    #                          'Boots_Hostility_dataset_mean', 'Boots_kDN_dataset_mean', 'Boots_DCP_dataset_mean',
    #                          'Boots_TD_U_dataset_mean', 'Boots_CLD_dataset_mean', 'Boots_N1_dataset_mean',
    #                          'Boots_N2_dataset_mean', 'Boots_LSC_dataset_mean', 'Boots_F1_dataset_mean']
    # df_aggre_std = results.groupby(['n_ensemble', 'weights'], as_index=False)[cols_numeric].std()
    # df_aggre_std.columns = ['n_ensemble', 'weights', 'accuracy_std', 'Boots_Hostility_dataset_std',
    #                         'Boots_kDN_dataset_std', 'Boots_DCP_dataset_std',
    #                         'Boots_TD_U_dataset_std', 'Boots_CLD_dataset_std', 'Boots_N1_dataset_std',
    #                         'Boots_N2_dataset_std', 'Boots_LSC_dataset_std', 'Boots_F1_dataset_std']
    #
    # df_aggre = pd.concat([df_aggre_mean, df_aggre_std.iloc[:, 2:]], axis=1)
    #
    # n_df = df_aggre.shape[0]
    # cols_names = ['confusion_matrix',
    #               'Boots_Hostility_class_mean', 'Boots_kDN_class_mean', 'Boots_DCP_class_mean',
    #               'Boots_TD_U_class_mean', 'Boots_CLD_class_mean', 'Boots_N1_class_mean',
    #               'Boots_N2_class_mean', 'Boots_LSC_class_mean', 'Boots_F1_class_mean',
    #               'Boots_Hostility_class_std', 'Boots_kDN_class_std', 'Boots_DCP_class_std',
    #               'Boots_TD_U_class_std', 'Boots_CLD_class_std', 'Boots_N1_class_std',
    #               'Boots_N2_class_std', 'Boots_LSC_class_std', 'Boots_F1_class_std']
    # df_lists = pd.DataFrame(0, index=np.arange(n_df), columns=cols_names)
    # df_aggre = pd.concat([df_aggre, df_lists], axis=1)
    #
    # n_ensemble_list = np.unique(results['n_ensemble']).tolist()
    # weights_list = np.unique(results['weights']).tolist()
    #
    # for n_i in n_ensemble_list:
    #     print(n_i)
    #     for w in weights_list:
    #         print(w)
    #         condition = (results.n_ensemble == n_i) & (results.weights == w)
    #         condition2 = (df_aggre.n_ensemble == n_i) & (df_aggre.weights == w)
    #         data_pack = results.loc[condition]
    #         conf_list = np.array(data_pack['confusion_matrix'].tolist())  # format
    #         conf_fold = np.sum(conf_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'confusion_matrix'] = str(conf_fold)
    #
    #         Host_class_list = np.array(data_pack['Boots_Hostility_class'].tolist())
    #         Host_class_fold_mean = np.mean(Host_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_Hostility_class_mean'] = str(Host_class_fold_mean)
    #         Host_class_fold_std = np.std(Host_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_Hostility_class_std'] = str(Host_class_fold_std)
    #
    #         kdn_class_list = np.array(data_pack['Boots_kDN_class'].tolist())
    #         kdn_class_fold_mean = np.mean(kdn_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_kDN_class_mean'] = str(kdn_class_fold_mean)
    #         kdn_class_fold_std = np.std(kdn_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_kDN_class_std'] = str(kdn_class_fold_std)
    #
    #         dcp_class_list = np.array(data_pack['Boots_DCP_class'].tolist())
    #         dcp_class_fold_mean = np.mean(dcp_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_DCP_class_mean'] = str(dcp_class_fold_mean)
    #         dcp_class_fold_std = np.std(dcp_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_DCP_class_std'] = str(dcp_class_fold_std)
    #
    #         tdu_class_list = np.array(data_pack['Boots_TD_U_class'].tolist())
    #         tdu_class_fold_mean = np.mean(tdu_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_TD_U_class_mean'] = str(tdu_class_fold_mean)
    #         tdu_class_fold_std = np.std(tdu_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_TD_U_class_std'] = str(tdu_class_fold_std)
    #
    #         cld_class_list = np.array(data_pack['Boots_CLD_class'].tolist())
    #         cld_class_fold_mean = np.mean(cld_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_CLD_class_mean'] = str(cld_class_fold_mean)
    #         cld_class_fold_std = np.std(cld_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_CLD_class_std'] = str(cld_class_fold_std)
    #
    #         n1_class_list = np.array(data_pack['Boots_N1_class'].tolist())
    #         n1_class_fold_mean = np.mean(n1_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_N1_class_mean'] = str(n1_class_fold_mean)
    #         n1_class_fold_std = np.std(n1_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_N1_class_std'] = str(n1_class_fold_std)
    #
    #         n2_class_list = np.array(data_pack['Boots_N2_class'].tolist())
    #         n2_class_fold_mean = np.mean(n2_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_N2_class_mean'] = str(n2_class_fold_mean)
    #         n2_class_fold_std = np.std(n2_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_N2_class_std'] = str(n2_class_fold_std)
    #
    #         lsc_class_list = np.array(data_pack['Boots_LSC_class'].tolist())
    #         lsc_class_fold_mean = np.mean(lsc_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_LSC_class_mean'] = str(lsc_class_fold_mean)
    #         lsc_class_fold_std = np.std(lsc_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_LSC_class_std'] = str(lsc_class_fold_std)
    #
    #         f1_class_list = np.array(data_pack['Boots_F1_class'].tolist())
    #         f1_class_fold_mean = np.mean(f1_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_F1_class_mean'] = str(f1_class_fold_mean)
    #         f1_class_fold_std = np.std(f1_class_list, axis=0).tolist()
    #         df_aggre.loc[condition2, 'Boots_F1_class_std'] = str(f1_class_fold_std)

    df_aggre = aggregation_results(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_Bagging_' + name_data + '_MoreWeight_' + emphasis + '_Instances.csv'
    df_aggre.to_csv(nombre_csv_agg, encoding='utf_8_sig',index=False)

    return results


path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


total_name_list = [#'teaching_assistant_MH.csv','contraceptive_NL.csv','hill_valley_without_noise_traintest.csv',
 # 'breast-w.csv','contraceptive_LS.csv','ilpd.csv','phoneme.csv',
 # 'mammographic.csv','contraceptive_NS.csv','bupa.csv','Yeast_CYTvsNUC.csv',
 # 'titanic.csv','arrhythmia_cfs.csv','vertebral_column.csv','sonar.csv',
 # 'spect_heart.csv','credit-g.csv', 'segment.csv',
                   'diabetes.csv',
 'diabetic_retinopathy.csv','WineQualityRed_5vs6.csv','teaching_assistant_LM.csv',
 'ionosphere.csv','bands.csv','wdbc.csv','teaching_assistant_LH.csv',
 'pima.csv','spambase.csv','banknote_authentication.csv', 'haberman.csv']
# 'appendicitis.csv' me ha dado problemas
total_name_list = ['Data4.csv']

path_to_save = root_path+'/Bagging_results'
n_ensembles = 200 # maximum number of ensembles to consider (later we plot and stop when we want)
# CM_selected = 'Hostility' # selection of the complexity measure to guide the sampling
#
for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data.iloc[:,:-1].to_numpy() # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy()
    # print(X.shape)

    emphasis_easy = 'easy'
    results = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_easy)
    emphasis_hard = 'hard'
    results2 = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_hard)
    # emphasis_easy3 = 'classes_easy'
    # results3 = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_easy3)
    # emphasis_hard4 = 'classes_hard'
    # results4 = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_hard4)
    # emphasis_frontier = 'frontier'
    # results5 = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_frontier)



# stump = 'yes'
# n_ensembles = 50
def complexity_driven_bagging_combo(X,y,n_ensembles, name_data,path_to_save, emphasis,stump):

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy',
                                    'Boots_Hostility_dataset','Boots_kDN_dataset','Boots_DCP_dataset',
                                    'Boots_TD_U_dataset','Boots_CLD_dataset', 'Boots_N1_dataset',
                                    'Boots_N2_dataset','Boots_LSC_dataset','Boots_F1_dataset',
                                    'Boots_Hostility_class', 'Boots_kDN_class','Boots_DCP_class',
                                    'Boots_TD_U_class','Boots_CLD_class','Boots_N1_class',
                                    'Boots_N2_class','Boots_LSC_class','Boots_F1_class'])

    # Complexity measures list to check
    CM_list = ['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1','Uniform']
    # CM_list = ['L1']
    # CM_list = ['Hostility', 'kDN', 'TD_U', 'N1', 'N2','LSC','F1','Uniform']

    # CM_selected = 'DCP'

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



        for CM_selected in CM_list:
            # print(CM_selected)

            if (CM_selected == 'Uniform'): # classic Bagging with uniform probability sampling
                weights = np.repeat(1/len(y_train), len(y_train), axis=0)
            else: # Sampling using Complexity measures
                CM_values = df_measures[CM_selected]
                if (emphasis == 'combo'):
                    ranking1 = CM_values.rank(method='average', ascending=True)  # more weight to difficult
                    ranking2 = CM_values.rank(method='average', ascending=False)  # more weight to easy

                elif (emphasis == 'combo_extreme'):
                    ranking1 = CM_values.rank(method='average', ascending=True)  # more weight to difficult
                    quantiles = np.quantile(ranking1, q=np.arange(0.5, 0.76, 0.25))
                    q50 = quantiles[0]
                    q75 = quantiles[1]
                    ranking1[(ranking1 >= q75)] = ranking1[(ranking1 >= q75)] * 4
                    ranking1[(ranking1 >= q50) & (ranking1 < q75)] = ranking1[(ranking1 >= q50) & (ranking1 < q75)] * 2

                    ranking2 = CM_values.rank(method='average', ascending=False)  # more weight to easy
                    quantiles_easy = np.quantile(ranking2, q=np.arange(0.5, 0.76, 0.25))
                    q50_easy = quantiles_easy[0]
                    q75_easy = quantiles_easy[1]
                    ranking2[(ranking2 >= q75_easy)] = ranking2[(ranking2 >= q75_easy)] * 4
                    ranking2[(ranking2 >= q50_easy) & (ranking2 < q75_easy)] = ranking2[(ranking2 >= q50_easy) & (
                                ranking2 < q75_easy)] * 2

                elif (emphasis == 'combo_classes'):
                    ## If we make per class specifically: ranking
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking1 = np.zeros(len(y_train_aux))
                    ranking_aux2 = np.zeros(len(y_train_aux))
                    ranking2 = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=True)
                        ranking1[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                        # more weight to easy
                        ranking_aux2[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=False)
                        ranking2[y_train_aux == c] = ranking_aux2[y_train_aux == c] / sum(ranking_aux2[y_train_aux == c])  # probability distribution
                elif (emphasis == 'combo_classes_extreme'):
                    ## If we make per class specifically: ranking
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking1 = np.zeros(len(y_train_aux))
                    ranking_aux2 = np.zeros(len(y_train_aux))
                    ranking2 = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=True)
                        quantiles = np.quantile(ranking_aux, q=np.arange(0.5, 0.76, 0.25))
                        q50 = quantiles[0]
                        q75 = quantiles[1]
                        ranking_aux[(ranking_aux >= q75)] = ranking_aux[(ranking_aux >= q75)] * 4
                        ranking_aux[(ranking_aux >= q50) & (ranking_aux < q75)] = ranking_aux[
                                                                             (ranking_aux >= q50) & (ranking_aux < q75)] * 2
                        ranking1[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                        # more weight to easy
                        ranking_aux2[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=False)
                        quantiles_easy = np.quantile(ranking_aux2, q=np.arange(0.5, 0.76, 0.25))
                        q50_easy = quantiles_easy[0]
                        q75_easy = quantiles_easy[1]
                        ranking_aux2[(ranking_aux2 >= q75_easy)] = ranking_aux2[(ranking_aux2 >= q75_easy)] * 4
                        ranking_aux2[(ranking_aux2 >= q50_easy) & (ranking_aux2 < q75_easy)] = ranking_aux2[(ranking_aux2 >= q50_easy) & (
                                ranking_aux2 < q75_easy)] * 2
                        ranking2[y_train_aux == c] = ranking_aux2[y_train_aux == c] / sum(ranking_aux2[y_train_aux == c])  # probability distribution


                weights = ranking1/sum(ranking1) # probability distribution
                weights2 = ranking2 / sum(ranking2)  # probability distribution

            preds = pd.DataFrame()
            ensemble_preds = pd.DataFrame()
            # i = 34
            for i in range(n_ensembles):

                # print(i)
                # Get bootstrap sample following CM_weights
                n_train = len(y_train)

                if (CM_selected == 'Uniform'):
                    np.random.seed(1)
                    X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)
                else:
                    if (i % 2 == 0): # even
                        # more weight to easy
                        np.random.seed(1)
                        X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights2)
                    else: # odd
                        # more weight to hard
                        np.random.seed(1)
                        X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)


                # Save complexity information (class and dataset levels)
                df_measures_bootstrap = df_measures_sel.iloc[bootstrap_indices].copy()
                # Binarize hostility to obtain class and dataset levels
                df_measures_bootstrap.loc[:, 'Hostility_bin'] = np.where(df_measures_bootstrap['Hostility'] >= 0.5, 1,0)

                # Dataset
                [Boots_kDN_dataset,Boots_DCP_dataset,Boots_TD_U_dataset,
                 Boots_CLD_dataset,Boots_N1_dataset,Boots_N2_dataset,
                 Boots_LSC_dataset,Boots_F1_dataset] = df_measures_bootstrap.mean()[1:-2]
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
                else: # Decision Stump
                    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
                clf.fit(X_bootstrap, y_bootstrap)
                y_pred = clf.predict(X_test)

                if (i==0): # first iteration
                    col_name = 'pred_' + str(i)
                    preds[col_name] = y_pred  # individual predictions
                    # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
                    y_predicted = y_pred
                else:
                    col_name = 'pred_'+str(i)
                    preds[col_name] = y_pred # individual predictions
                    votes = voting_rule(preds)
                    # print(votes)
                    votes_dict = {'col_name':votes}
                    votes_df = pd.DataFrame(votes_dict)
                    votes_df.columns = [col_name]
                    # ensemble_preds[col_name] = votes # ensemble prediction with majority voting rule
                    ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)
                    # print(ensemble_preds[col_name])

                    y_predicted = ensemble_preds.iloc[:, -1:] # last column
                acc = accuracy_score(y_predicted, y_test)
                conf_matrix = confusion_matrix(y_test, y_predicted).tolist()

                results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                                    'confusion_matrix':[conf_matrix], 'accuracy':acc,
                                    'Boots_Hostility_dataset':Boots_Hostility_dataset,
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
                results = pd.concat([results,results_aux])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Bagging_' + name_data + '_MoreWeight_' + emphasis + '_stump_' + stump+'Instances.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)

    ##### Agregation of results
    df_aggre = aggregation_results(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_Bagging_' + name_data + '_MoreWeight_' + emphasis + '_stump_' + stump+ 'Instances.csv'
    df_aggre.to_csv(nombre_csv_agg, encoding='utf_8_sig',index=False)

    return results



def complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis, split, stump):

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy',
                                    'Boots_Hostility_dataset','Boots_kDN_dataset','Boots_DCP_dataset',
                                    'Boots_TD_U_dataset','Boots_CLD_dataset', 'Boots_N1_dataset',
                                    'Boots_N2_dataset','Boots_LSC_dataset','Boots_F1_dataset',
                                    'Boots_Hostility_class', 'Boots_kDN_class','Boots_DCP_class',
                                    'Boots_TD_U_class','Boots_CLD_class','Boots_N1_class',
                                    'Boots_N2_class','Boots_LSC_class','Boots_F1_class'])

    # Complexity measures list to check
    # CM_list = ['Hostility', 'kDN', 'TD_U', 'N1', 'N2','LSC','F1','Uniform']
    CM_list = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1', 'Uniform']
    # CM_list = ['L1']
    # CM_selected = 'Hostility'

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

        for CM_selected in CM_list:
            # print(CM_selected)

            if (CM_selected == 'Uniform'): # classic Bagging with uniform probability sampling
                weights = np.repeat(1/len(y_train), len(y_train), axis=0)
            else: # Sampling using Complexity measures
                CM_values = df_measures[CM_selected]

                if (emphasis == 'combo_split'):
                    ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
                    ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
                    weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution
                    weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
                    w_frac = (weights_hard - weights_easy) / split
                    weights_v = pd.DataFrame()
                    for s in range(split+1):
                        # print(s)
                        new_w = weights_easy + s*w_frac
                        weights_v = pd.concat([weights_v,new_w],axis=1)
                elif (emphasis == 'combo_split_classic'):
                    # split = 1 --> easy-uniform-hard
                    ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
                    ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
                    weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution
                    weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
                    weights_classic = np.repeat(1 / len(y_train), len(y_train), axis=0)
                    w_frac1 = (weights_classic - weights_easy) / split
                    w_frac2 = (weights_hard - weights_classic) / split
                    weights_v = pd.DataFrame()
                    for s in range(split + 1):
                        # print(s)
                        new_w1 = weights_easy + s * w_frac1
                        new_w_df1 = pd.DataFrame(new_w1)
                        weights_v = pd.concat([weights_v, new_w_df1], axis=1)
                    for s in np.arange(1,split+1):
                        # print(s)
                        new_w2 = weights_classic + s * w_frac2
                        new_w_df2 = pd.DataFrame(new_w2)
                        weights_v = pd.concat([weights_v, new_w_df2], axis=1)

                elif (emphasis == 'combo_split_classes'):
                    ## If we make per class specifically: ranking
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking_hard = np.zeros(len(y_train_aux))
                    ranking_aux2 = np.zeros(len(y_train_aux))
                    ranking_easy = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=True)
                        ranking_hard[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution per class
                        # more weight to easy
                        ranking_aux2[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average',ascending=False)
                        ranking_easy[y_train_aux == c] = ranking_aux2[y_train_aux == c] / sum(ranking_aux2[y_train_aux == c])  # probability distribution per class

                    weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution global
                    weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution global
                    w_frac = (weights_hard - weights_easy) / split
                    weights_v = pd.DataFrame()
                    for s in range(split+1):
                        # print(s)
                        new_w = weights_easy + s*w_frac
                        new_w_df = pd.DataFrame(new_w)
                        weights_v = pd.concat([weights_v,new_w_df],axis=1)
                elif (emphasis == 'combo_split_extreme'):
                    ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
                    quantiles = np.quantile(ranking_hard, q=np.arange(0.5, 0.76, 0.25))
                    q50 = quantiles[0]
                    q75 = quantiles[1]
                    ranking_hard[(ranking_hard >= q75)] = ranking_hard[(ranking_hard >= q75)] * 4
                    ranking_hard[(ranking_hard >= q50) & (ranking_hard < q75)] = ranking_hard[(ranking_hard >= q50) & (
                                ranking_hard < q75)] * 2

                    ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
                    quantiles_easy = np.quantile(ranking_easy, q=np.arange(0.5, 0.76, 0.25))
                    q50_easy = quantiles_easy[0]
                    q75_easy = quantiles_easy[1]
                    ranking_easy[(ranking_easy >= q75_easy)] = ranking_easy[(ranking_easy >= q75_easy)] * 4
                    ranking_easy[(ranking_easy >= q50_easy) & (ranking_easy < q75_easy)] = ranking_easy[(
                                                                                                                    ranking_easy >= q50_easy) & (
                                                                                                                ranking_easy < q75_easy)] * 2

                    weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution
                    weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
                    w_frac = (weights_hard - weights_easy) / split
                    weights_v = pd.DataFrame()
                    for s in range(split + 1):
                        # print(s)
                        new_w = weights_easy + s * w_frac
                        new_w_df = pd.DataFrame(new_w)
                        weights_v = pd.concat([weights_v, new_w_df], axis=1)
                elif (emphasis == 'combo_split_classic_extreme'):
                    ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
                    quantiles = np.quantile(ranking_hard, q=np.arange(0.5, 0.76, 0.25))
                    q50 = quantiles[0]
                    q75 = quantiles[1]
                    ranking_hard[(ranking_hard >= q75)] = ranking_hard[(ranking_hard >= q75)] * 4
                    ranking_hard[(ranking_hard >= q50) & (ranking_hard < q75)] = ranking_hard[(ranking_hard >= q50) & (
                                ranking_hard < q75)] * 2

                    ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
                    quantiles_easy = np.quantile(ranking_easy, q=np.arange(0.5, 0.76, 0.25))
                    q50_easy = quantiles_easy[0]
                    q75_easy = quantiles_easy[1]
                    ranking_easy[(ranking_easy >= q75_easy)] = ranking_easy[(ranking_easy >= q75_easy)] * 4
                    ranking_easy[(ranking_easy >= q50_easy) & (ranking_easy < q75_easy)] = ranking_easy[(
                                                                                                                    ranking_easy >= q50_easy) & (
                                                                                                                ranking_easy < q75_easy)] * 2

                    weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution
                    weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
                    weights_classic = np.repeat(1 / len(y_train), len(y_train), axis=0)
                    w_frac1 = (weights_classic - weights_easy) / split
                    w_frac2 = (weights_hard - weights_classic) / split
                    weights_v = pd.DataFrame()
                    for s in range(split + 1):
                        # print(s)
                        new_w1 = weights_easy + s * w_frac1
                        new_w_df1 = pd.DataFrame(new_w1)
                        weights_v = pd.concat([weights_v, new_w_df1], axis=1)
                    for s in np.arange(1,split+1):
                        print(s)
                        new_w2 = weights_classic + s * w_frac2
                        new_w_df2 = pd.DataFrame(new_w2)
                        weights_v = pd.concat([weights_v, new_w_df2], axis=1)

                elif (emphasis == 'combo_split_extreme_classes'):
                    ## If we make per class specifically: ranking
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking_hard = np.zeros(len(y_train_aux))
                    ranking_aux2 = np.zeros(len(y_train_aux))
                    ranking_easy = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average', ascending=True)
                        quantiles = np.quantile(ranking_aux, q=np.arange(0.5, 0.76, 0.25))
                        q50 = quantiles[0]
                        q75 = quantiles[1]
                        ranking_aux[(ranking_aux >= q75)] = ranking_aux[(ranking_aux >= q75)] * 4
                        ranking_aux[(ranking_aux >= q50) & (ranking_aux < q75)] = ranking_aux[
                                                                                      (ranking_aux >= q50) & (
                                                                                                  ranking_aux < q75)] * 2
                        ranking_hard[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(
                            ranking_aux[y_train_aux == c])  # probability distribution
                        # more weight to easy
                        ranking_aux2[y_train_aux == c] = CM_values[y_train_aux == c].rank(method='average', ascending=False)
                        quantiles_easy = np.quantile(ranking_aux2, q=np.arange(0.5, 0.76, 0.25))
                        q50_easy = quantiles_easy[0]
                        q75_easy = quantiles_easy[1]
                        ranking_aux2[(ranking_aux2 >= q75_easy)] = ranking_aux2[(ranking_aux2 >= q75_easy)] * 4
                        ranking_aux2[(ranking_aux2 >= q50_easy) & (ranking_aux2 < q75_easy)] = ranking_aux2[(
                                                                                                                        ranking_aux2 >= q50_easy) & (
                                                                                                                    ranking_aux2 < q75_easy)] * 2
                        ranking_easy[y_train_aux == c] = ranking_aux2[y_train_aux == c] / sum(
                            ranking_aux2[y_train_aux == c])  # probability distribution

                    weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution
                    weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
                    w_frac = (weights_hard - weights_easy) / split
                    weights_v = pd.DataFrame()
                    for s in range(split + 1):
                        # print(s)
                        new_w = weights_easy + s * w_frac
                        new_w_df = pd.DataFrame(new_w)
                        weights_v = pd.concat([weights_v, new_w_df], axis=1)


            preds = pd.DataFrame()
            ensemble_preds = pd.DataFrame()
            # i = 0
            j = 0
            for i in range(n_ensembles):

                # print(i)
                # Get bootstrap sample following CM_weights
                n_train = len(y_train)

                if (CM_selected == 'Uniform'):
                    np.random.seed(1)
                    X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)
                else:
                    index_split = weights_v.shape[1] - 1
                    if (j <= index_split):
                        weights = weights_v.iloc[:,j]
                    else:
                        j = 0
                        weights = weights_v.iloc[:, j]
                    # print('j=', j)
                    j = j + 1


                    np.random.seed(1)
                    X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)


                # Save complexity information (class and dataset levels)
                df_measures_bootstrap = df_measures_sel.iloc[bootstrap_indices].copy()
                # Binarize hostility to obtain class and dataset levels
                df_measures_bootstrap.loc[:, 'Hostility_bin'] = np.where(df_measures_bootstrap['Hostility'] >= 0.5, 1,0)

                # Dataset
                [Boots_kDN_dataset,Boots_DCP_dataset,Boots_TD_U_dataset,
                 Boots_CLD_dataset,Boots_N1_dataset,Boots_N2_dataset,
                 Boots_LSC_dataset,Boots_F1_dataset] = df_measures_bootstrap.mean()[1:-2]
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

                if (i==0): # first iteration
                    col_name = 'pred_' + str(i)
                    preds[col_name] = y_pred  # individual predictions
                    # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
                    y_predicted = y_pred
                else:
                    col_name = 'pred_'+str(i)
                    preds[col_name] = y_pred # individual predictions
                    votes = voting_rule(preds)
                    # print(votes)
                    votes_dict = {'col_name':votes}
                    votes_df = pd.DataFrame(votes_dict)
                    votes_df.columns = [col_name]
                    # ensemble_preds[col_name] = votes # ensemble prediction with majority voting rule
                    ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)
                    # print(ensemble_preds[col_name])

                    y_predicted = ensemble_preds.iloc[:, -1:] # last column
                acc = accuracy_score(y_predicted, y_test)
                conf_matrix = confusion_matrix(y_test, y_predicted).tolist()

                results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                                    'confusion_matrix':[conf_matrix], 'accuracy':acc,
                                    'Boots_Hostility_dataset':Boots_Hostility_dataset,
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
                results = pd.concat([results,results_aux])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Bagging_' + name_data + '_MoreWeight_' + emphasis + '_split' + str(split)+ '_stump_' + stump+ 'Instances.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)

    ##### Agregation of results
    df_aggre = aggregation_results(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_Bagging_' + name_data + '_MoreWeight_' + emphasis + '_split' + str(split) + '_stump_' + stump + 'Instances.csv'
    df_aggre.to_csv(nombre_csv_agg, encoding='utf_8_sig',index=False)

    return results





path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)

# yeast da problemas porque una clase es muy pequeña y no aparece en todos los folds (creo que tb es por DCP)
# haberman da problemas y es por DCP que da solo dos valores y concuerdan con la y

# total_name_list = ['Data13.csv']

total_name_list = [#'teaching_assistant_MH.csv','contraceptive_NL.csv','hill_valley_without_noise_traintest.csv',
 # 'breast-w.csv','contraceptive_LS.csv','ilpd.csv','phoneme.csv',
 # 'mammographic.csv','contraceptive_NS.csv','bupa.csv','Yeast_CYTvsNUC.csv',
 # 'titanic.csv','arrhythmia_cfs.csv','vertebral_column.csv','sonar.csv',
 #'spect_heart.csv',
 #                   'credit-g.csv', 'segment.csv',
 #                   ##'appendicitis.csv', 'haberman.csv',
 #                   'diabetes.csv',
 # 'diabetic_retinopathy.csv','WineQualityRed_5vs6.csv','teaching_assistant_LM.csv',
'teaching_assistant_LH.csv',
 'ionosphere.csv','bands.csv','wdbc.csv',
    'spambase.csv','banknote_authentication.csv', 'pima.csv','titanic.csv']
# 'appendicitis.csv' y haberman me han dado problemas
total_name_list = ['titanic.csv']

total_name_list = ['Data1.csv','Data2.csv','Data3.csv','Data4.csv','Data5.csv',
                'Data6.csv','Data7.csv', 'Data8.csv','Data9.csv','Data10.csv',
                'Data11.csv','Data12.csv',  'Data13.csv']

path_to_save = root_path+'/Bagging_results'
n_ensembles = 200 # maximum number of ensembles to consider (later we plot and stop when we want)
# CM_selected = 'Hostility' # selection of the complexity measure to guide the sampling

for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data.iloc[:,:-1].to_numpy() # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy()
    stump = 'no'
    split1 = 1
    emphasis0_cl = 'combo_split_classic' # split = 1 es easy-uniform-hard
    results0_cl = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis0_cl, split1, stump)
    emphasis1 = 'combo_split_classic_extreme'
    results1 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data, path_to_save, emphasis1, split1,stump)
    split2 = 2 # with combo_split_classic this is 5 splits
    results2 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis0_cl, split2, stump)
    results3 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data, path_to_save, emphasis1, split2,stump)
    split4 = 4 # with combo_split_classic this is 9 splits
    results4 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis0_cl, split4, stump)
    results5 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data, path_to_save, emphasis1, split4,stump)

    emphasis0 = 'combo_extreme'
    results0 = complexity_driven_bagging_combo(X, y, n_ensembles, name_data, path_to_save, emphasis0, stump)
    emphasis00 = 'combo'
    results00 = complexity_driven_bagging_combo(X, y, n_ensembles, name_data, path_to_save, emphasis00, stump)
    emphasis = 'combo_split'
    split = 2
    results1 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis, split, stump)
    split4 = 4
    results2 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis, split4, stump)
    split9 = 9
    results3 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis, split9, stump)
    emphasis2 = 'combo_split_extreme'
    split = 2
    results4 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis2, split, stump)
    split4 = 4
    results5 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis2, split4, stump)
    split9 = 9
    results6 = complexity_driven_bagging_combo_split(X,y,n_ensembles, name_data,path_to_save, emphasis2, split9, stump)

    emphasis_easy = 'easy'
    results_easy = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_easy)
    emphasis_hard = 'hard'
    results_hard = complexity_driven_bagging(X, y, n_ensembles, name_data, path_to_save,emphasis_hard)









def complexity_driven_bagging_averaged(X,y,n_ensembles, name_data,path_to_save, emphasis):

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy',
                                    'Boots_Hostility_dataset','Boots_kDN_dataset','Boots_DCP_dataset',
                                    'Boots_TD_U_dataset','Boots_CLD_dataset', 'Boots_N1_dataset',
                                    'Boots_N2_dataset','Boots_LSC_dataset','Boots_F1_dataset',
                                    'Boots_Hostility_class', 'Boots_kDN_class','Boots_DCP_class',
                                    'Boots_TD_U_class','Boots_CLD_class','Boots_N1_class',
                                    'Boots_N2_class','Boots_LSC_class','Boots_F1_class'])

    # Complexity measures list to check
    # CM_list = ['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1','Uniform']
    CM_list = ['Averaged_measures', 'Uniform']
    # CM_selected = 'F1'

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
        data_train = pd.DataFrame(X_train, columns=['x1','x2'])
        data_train['y'] = y_train
        df_measures, _ = all_measures(data_train,False,None, None)
        # Selection of complexity measures
        df_measures_sel = df_measures[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1','y']]

        for CM_selected in CM_list:
            print(CM_selected)

            if (CM_selected == 'Uniform'): # classic Bagging with uniform probability sampling
                weights = np.repeat(1/len(y_train), len(y_train), axis=0)
            else: # Sampling using averaged ranking of Complexity measures
                if (emphasis == 'averaged_hard'):
                    # more weight to difficult
                    ranking = df_measures_sel[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1']].rank(method='average', ascending=True).mean(axis=1)
                elif (emphasis == 'averaged_easy'):
                    # more weight to easy
                    ranking = df_measures_sel[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1']].rank(method='average', ascending=False).mean(axis=1)
                # elif (emphasis == 'combo'):
                #     # more weight to difficult
                #     ranking1 = df_measures_sel[['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']].rank(
                #         method='average', ascending=True).mean(axis=1)
                #     # more weight to easy
                #     ranking2 = df_measures_sel[['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']].rank(
                #         method='average', ascending=False).mean(axis=1)
                elif (emphasis == 'averaged_classes_hard'):
                    ## If we make per class specifically: ranking
                    df_sel_class = df_measures_sel[['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']]
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = df_sel_class[y_train_aux == c].rank(method='average', ascending=True).mean(axis=1)
                        ranking[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(
                            ranking_aux[y_train_aux == c])  # probability distribution
                elif (emphasis == 'averaged_classes_easy'):
                    ## If we make per class specifically: ranking
                    df_sel_class = df_measures_sel[['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']]
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to easy
                        ranking_aux[y_train_aux == c] = df_sel_class[y_train_aux == c].rank(method='average', ascending=False).mean(axis=1)
                        ranking[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(
                            ranking_aux[y_train_aux == c])  # probability distribution


                weights = ranking/sum(ranking) # probability distribution
                # weights2 = ranking2 / sum(ranking2)  # probability distribution

            preds = pd.DataFrame()
            ensemble_preds = pd.DataFrame()
            # i = 0
            for i in range(n_ensembles):

                # print(i)
                # Get bootstrap sample following CM_weights
                n_train = len(y_train)

                np.random.seed(1)
                X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)


                # Save complexity information (class and dataset levels)
                df_measures_bootstrap = df_measures_sel.iloc[bootstrap_indices].copy()
                # Binarize hostility to obtain class and dataset levels
                df_measures_bootstrap.loc[:, 'Hostility_bin'] = np.where(df_measures_bootstrap['Hostility'] >= 0.5, 1,0)

                # Dataset
                [Boots_kDN_dataset,Boots_DCP_dataset,Boots_TD_U_dataset,
                 Boots_CLD_dataset,Boots_N1_dataset,Boots_N2_dataset,
                 Boots_LSC_dataset,Boots_F1_dataset] = df_measures_bootstrap.mean()[1:-2]
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
                clf = DecisionTreeClassifier(random_state=0)
                clf.fit(X_bootstrap, y_bootstrap)
                y_pred = clf.predict(X_test)

                if (i==0): # first iteration
                    col_name = 'pred_' + str(i)
                    preds[col_name] = y_pred  # individual predictions
                    # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
                    y_predicted = y_pred
                else:
                    col_name = 'pred_'+str(i)
                    preds[col_name] = y_pred # individual predictions
                    votes = voting_rule(preds)
                    # print(votes)
                    votes_dict = {'col_name':votes}
                    votes_df = pd.DataFrame(votes_dict)
                    votes_df.columns = [col_name]
                    # ensemble_preds[col_name] = votes # ensemble prediction with majority voting rule
                    ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)
                    # print(ensemble_preds[col_name])

                    y_predicted = ensemble_preds.iloc[:, -1:] # last column
                acc = accuracy_score(y_predicted, y_test)
                conf_matrix = confusion_matrix(y_test, y_predicted).tolist()

                results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                                    'confusion_matrix':[conf_matrix], 'accuracy':acc,
                                    'Boots_Hostility_dataset':Boots_Hostility_dataset,
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
                results = pd.concat([results,results_aux])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Bagging_' + name_data + '_MoreWeight_' + emphasis +'_Instances.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)

    ##### Agregation of results
    df_aggre = aggregation_results(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_Bagging_' + name_data + '_MoreWeight_' + emphasis + '_Instances.csv'
    df_aggre.to_csv(nombre_csv_agg, encoding='utf_8_sig',index=False)

    return results



path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


# total_name_list = ['Data13.csv']

path_to_save = root_path+'/Bagging_results'
n_ensembles = 200 # maximum number of ensembles to consider (later we plot and stop when we want)
# CM_selected = 'Hostility' # selection of the complexity measure to guide the sampling

for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data[['x1', 'x2']].to_numpy()
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy()
    emphasis_easy = 'averaged_easy'
    results = complexity_driven_bagging_averaged(X, y, n_ensembles, name_data, path_to_save,emphasis_easy)
    emphasis_hard = 'averaged_hard'
    results2 = complexity_driven_bagging_averaged(X, y, n_ensembles, name_data, path_to_save,emphasis_hard)
    emphasis_easy_class = 'averaged_classes_easy'
    results3 = complexity_driven_bagging_averaged(X, y, n_ensembles, name_data, path_to_save,emphasis_easy_class)
    emphasis_hard_class = 'averaged_classes_hard'
    results4 = complexity_driven_bagging_averaged(X, y, n_ensembles, name_data, path_to_save,emphasis_hard_class)





def complexity_driven_bagging_averaged_combo(X,y,n_ensembles, name_data,path_to_save, emphasis):

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy',
                                    'Boots_Hostility_dataset','Boots_kDN_dataset','Boots_DCP_dataset',
                                    'Boots_TD_U_dataset','Boots_CLD_dataset', 'Boots_N1_dataset',
                                    'Boots_N2_dataset','Boots_LSC_dataset','Boots_F1_dataset',
                                    'Boots_Hostility_class', 'Boots_kDN_class','Boots_DCP_class',
                                    'Boots_TD_U_class','Boots_CLD_class','Boots_N1_class',
                                    'Boots_N2_class','Boots_LSC_class','Boots_F1_class'])

    # Complexity measures list to check
    # CM_list = ['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1','Uniform']
    CM_list = ['Averaged_measures', 'Uniform']
    # CM_selected = 'F1'

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
        data_train = pd.DataFrame(X_train, columns=['x1','x2'])
        data_train['y'] = y_train
        df_measures, _ = all_measures(data_train,False,None, None)
        # Selection of complexity measures
        df_measures_sel = df_measures[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1','y']]

        for CM_selected in CM_list:
            print(CM_selected)

            if (CM_selected == 'Uniform'): # classic Bagging with uniform probability sampling
                weights = np.repeat(1/len(y_train), len(y_train), axis=0)
            else: # Sampling using averaged ranking of Complexity measures
                if (emphasis == 'averaged_combo'):
                    # more weight to difficult
                    ranking1 = df_measures_sel[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1']].rank(method='average', ascending=True).mean(axis=1)
                    # more weight to easy
                    ranking2 = df_measures_sel[['Hostility', 'kDN', 'DCP','TD_U', 'CLD', 'N1', 'N2','LSC','F1']].rank(method='average', ascending=False).mean(axis=1)
                elif (emphasis == 'averaged_combo_classes'):
                    ## If we make per class specifically: ranking
                    df_sel_class = df_measures_sel[['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']]
                    y_train_aux = np.concatenate(y_train, axis=0)
                    n_classes = len(np.unique(y_train))
                    ranking_aux = np.zeros(len(y_train_aux))
                    ranking1 = np.zeros(len(y_train_aux))
                    ranking_aux2 = np.zeros(len(y_train_aux))
                    ranking2 = np.zeros(len(y_train_aux))
                    for c in range(n_classes):
                        # print(c)
                        n_class_c = np.sum(y_train_aux == c)
                        # more weight to difficult
                        ranking_aux[y_train_aux == c] = df_sel_class[y_train_aux == c].rank(method='average', ascending=True).mean(axis=1)
                        ranking1[y_train_aux == c] = ranking_aux[y_train_aux == c] / sum(ranking_aux[y_train_aux == c])  # probability distribution
                        # more weight to easy
                        ranking_aux2[y_train_aux == c] = df_sel_class[y_train_aux == c].rank(method='average', ascending=False).mean(axis=1)
                        ranking2[y_train_aux == c] = ranking_aux2[y_train_aux == c] / sum(ranking_aux2[y_train_aux == c])  # probability distribution

                weights = ranking1/sum(ranking1) # probability distribution
                weights2 = ranking2 / sum(ranking2)  # probability distribution

            preds = pd.DataFrame()
            ensemble_preds = pd.DataFrame()
            # i = 0
            for i in range(n_ensembles):

                # print(i)
                # Get bootstrap sample following CM_weights
                n_train = len(y_train)

                if (CM_selected == 'Uniform'):
                    np.random.seed(1)
                    X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)
                else:
                    if (i % 2 == 0): # even
                        # more weight to easy
                        np.random.seed(1)
                        X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights2)
                    else: # odd
                        # more weight to hard
                        np.random.seed(1)
                        X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)


                # Save complexity information (class and dataset levels)
                df_measures_bootstrap = df_measures_sel.iloc[bootstrap_indices].copy()
                # Binarize hostility to obtain class and dataset levels
                df_measures_bootstrap.loc[:, 'Hostility_bin'] = np.where(df_measures_bootstrap['Hostility'] >= 0.5, 1,0)

                # Dataset
                [Boots_kDN_dataset,Boots_DCP_dataset,Boots_TD_U_dataset,
                 Boots_CLD_dataset,Boots_N1_dataset,Boots_N2_dataset,
                 Boots_LSC_dataset,Boots_F1_dataset] = df_measures_bootstrap.mean()[1:-2]
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
                clf = DecisionTreeClassifier(random_state=0)
                clf.fit(X_bootstrap, y_bootstrap)
                y_pred = clf.predict(X_test)

                if (i==0): # first iteration
                    col_name = 'pred_' + str(i)
                    preds[col_name] = y_pred  # individual predictions
                    # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
                    y_predicted = y_pred
                else:
                    col_name = 'pred_'+str(i)
                    preds[col_name] = y_pred # individual predictions
                    votes = voting_rule(preds)
                    # print(votes)
                    votes_dict = {'col_name':votes}
                    votes_df = pd.DataFrame(votes_dict)
                    votes_df.columns = [col_name]
                    # ensemble_preds[col_name] = votes # ensemble prediction with majority voting rule
                    ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)
                    # print(ensemble_preds[col_name])

                    y_predicted = ensemble_preds.iloc[:, -1:] # last column
                acc = accuracy_score(y_predicted, y_test)
                conf_matrix = confusion_matrix(y_test, y_predicted).tolist()

                results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                                    'confusion_matrix':[conf_matrix], 'accuracy':acc,
                                    'Boots_Hostility_dataset':Boots_Hostility_dataset,
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
                results = pd.concat([results,results_aux])

    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Bagging_' + name_data + '_MoreWeight_' + emphasis +'_Instances.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)

    ##### Agregation of results
    df_aggre = aggregation_results(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_Bagging_' + name_data + '_MoreWeight_' + emphasis + '_Instances.csv'
    df_aggre.to_csv(nombre_csv_agg, encoding='utf_8_sig',index=False)

    return results



path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


# total_name_list = ['Data13.csv']

path_to_save = root_path+'/Bagging_results'
n_ensembles = 200 # maximum number of ensembles to consider (later we plot and stop when we want)
# CM_selected = 'Hostility' # selection of the complexity measure to guide the sampling

for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data[['x1', 'x2']].to_numpy()
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy()
    emphasis_combo = 'averaged_combo'
    results = complexity_driven_bagging_averaged_combo(X, y, n_ensembles, name_data, path_to_save,emphasis_combo)
    emphasis_combo_class = 'averaged_combo_classes'
    results2 = complexity_driven_bagging_averaged_combo(X, y, n_ensembles, name_data, path_to_save,emphasis_combo_class)

















