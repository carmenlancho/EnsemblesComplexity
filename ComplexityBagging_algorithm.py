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





def ComplexityDrivenBagging(X,y,n_ensembles, name_data,path_to_save, split, stump,alpha):

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
    CM_list = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']
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
            CM_values = df_measures[CM_selected]

            ### combo split classic extreme
            ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
            ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
            if (alpha >=2):
                quantiles = np.quantile(ranking_hard, q=np.arange(0.5, 0.76, 0.25))
                q50 = quantiles[0]
                q75 = quantiles[1]
                ranking_hard[(ranking_hard >= q75)] = ranking_hard[(ranking_hard >= q75)] * alpha
                ranking_hard[(ranking_hard >= q50) & (ranking_hard < q75)] = ranking_hard[(ranking_hard >= q50) & (
                        ranking_hard < q75)] * (alpha/2)


                quantiles_easy = np.quantile(ranking_easy, q=np.arange(0.5, 0.76, 0.25))
                q50_easy = quantiles_easy[0]
                q75_easy = quantiles_easy[1]
                ranking_easy[(ranking_easy >= q75_easy)] = ranking_easy[(ranking_easy >= q75_easy)] * alpha
                ranking_easy[(ranking_easy >= q50_easy) & (ranking_easy < q75_easy)] = ranking_easy[(
                                                                                                            ranking_easy >= q50_easy) & (
                                                                                                            ranking_easy < q75_easy)] * (alpha/2)
            # if alpha < 2, then no extreme weights are applied


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
            for s in np.arange(1, split + 1):
                print(s)
                new_w2 = weights_classic + s * w_frac2
                new_w_df2 = pd.DataFrame(new_w2)
                weights_v = pd.concat([weights_v, new_w_df2], axis=1)

            preds = pd.DataFrame()
            ensemble_preds = pd.DataFrame()
            # i = 0
            j = 0
            for i in range(n_ensembles):

                # print(i)
                # Get bootstrap sample following CM_weights
                n_train = len(y_train)

                index_split = weights_v.shape[1] - 1
                if (j <= index_split):
                    weights = weights_v.iloc[:, j]
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
    nombre_csv = 'CDB_' + name_data + '_split' + str(split)+ '_alpha_' + str(alpha)+ '.csv'
    results.to_csv(nombre_csv, encoding='utf_8_sig',index=False)

    ##### Agregation of results
    df_aggre = aggregation_results_final_algorith(results)

    # To save the results
    os.chdir(path_to_save)
    nombre_csv_agg = 'AggregatedResults_CDB_' + name_data + '_split' + str(split)+ '_alpha_' + str(alpha)+ '.csv'
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
total_name_list = ['ionosphere.csv']

# total_name_list = ['Data1.csv','Data2.csv','Data3.csv','Data4.csv','Data5.csv',
#                 'Data6.csv','Data7.csv', 'Data8.csv','Data9.csv','Data10.csv',
#                 'Data11.csv','Data12.csv',  'Data13.csv']

path_to_save = root_path+'/Results_general_algorithm'
n_ensembles = 10 # maximum number of ensembles to consider (later we plot and stop when we want)
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
    results0 = ComplexityDrivenBagging(X,y,n_ensembles, name_data,path_to_save, split1, stump,alpha)




