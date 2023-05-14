import numpy as np
import pandas as pd

# results = results0
def aggregation_results(results):
    ##### Agregation of results
    cols_numeric = results.select_dtypes([np.number]).columns
    df_aggre_mean = results.groupby(['n_ensemble', 'weights'], as_index=False)[cols_numeric].mean()
    df_aggre_mean.columns = ['n_ensemble', 'weights', 'accuracy_mean',
                             'Boots_Hostility_dataset_mean', 'Boots_kDN_dataset_mean', 'Boots_DCP_dataset_mean',
                             'Boots_TD_U_dataset_mean', 'Boots_CLD_dataset_mean', 'Boots_N1_dataset_mean',
                             'Boots_N2_dataset_mean', 'Boots_LSC_dataset_mean', 'Boots_F1_dataset_mean']
    df_aggre_std = results.groupby(['n_ensemble', 'weights'], as_index=False)[cols_numeric].std()
    df_aggre_std.columns = ['n_ensemble', 'weights', 'accuracy_std', 'Boots_Hostility_dataset_std',
                            'Boots_kDN_dataset_std', 'Boots_DCP_dataset_std',
                            'Boots_TD_U_dataset_std', 'Boots_CLD_dataset_std', 'Boots_N1_dataset_std',
                            'Boots_N2_dataset_std', 'Boots_LSC_dataset_std', 'Boots_F1_dataset_std']

    df_aggre = pd.concat([df_aggre_mean, df_aggre_std.iloc[:, 2:]], axis=1)

    n_df = df_aggre.shape[0]
    cols_names = ['confusion_matrix',
                  'Boots_Hostility_class_mean', 'Boots_kDN_class_mean', 'Boots_DCP_class_mean',
                  'Boots_TD_U_class_mean', 'Boots_CLD_class_mean', 'Boots_N1_class_mean',
                  'Boots_N2_class_mean', 'Boots_LSC_class_mean', 'Boots_F1_class_mean',
                  'Boots_Hostility_class_std', 'Boots_kDN_class_std', 'Boots_DCP_class_std',
                  'Boots_TD_U_class_std', 'Boots_CLD_class_std', 'Boots_N1_class_std',
                  'Boots_N2_class_std', 'Boots_LSC_class_std', 'Boots_F1_class_std']
    df_lists = pd.DataFrame(0, index=np.arange(n_df), columns=cols_names)
    df_aggre = pd.concat([df_aggre, df_lists], axis=1)

    n_ensemble_list = np.unique(results['n_ensemble']).tolist()
    weights_list = np.unique(results['weights']).tolist()

    for n_i in n_ensemble_list:
        print(n_i)
        for w in weights_list:
            print(w)
            condition = (results.n_ensemble == n_i) & (results.weights == w)
            condition2 = (df_aggre.n_ensemble == n_i) & (df_aggre.weights == w)
            data_pack = results.loc[condition]
            conf_list = np.array(data_pack['confusion_matrix'].tolist())  # format
            conf_fold = np.sum(conf_list, axis=0).tolist()
            df_aggre.loc[condition2, 'confusion_matrix'] = str(conf_fold)

            Host_class_list = np.array(data_pack['Boots_Hostility_class'].tolist())
            Host_class_fold_mean = np.mean(Host_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_Hostility_class_mean'] = str(Host_class_fold_mean)
            Host_class_fold_std = np.std(Host_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_Hostility_class_std'] = str(Host_class_fold_std)

            kdn_class_list = np.array(data_pack['Boots_kDN_class'].tolist())
            kdn_class_fold_mean = np.mean(kdn_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_kDN_class_mean'] = str(kdn_class_fold_mean)
            kdn_class_fold_std = np.std(kdn_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_kDN_class_std'] = str(kdn_class_fold_std)

            dcp_class_list = np.array(data_pack['Boots_DCP_class'].tolist())
            dcp_class_fold_mean = np.mean(dcp_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_DCP_class_mean'] = str(dcp_class_fold_mean)
            dcp_class_fold_std = np.std(dcp_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_DCP_class_std'] = str(dcp_class_fold_std)

            tdu_class_list = np.array(data_pack['Boots_TD_U_class'].tolist())
            tdu_class_fold_mean = np.mean(tdu_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_TD_U_class_mean'] = str(tdu_class_fold_mean)
            tdu_class_fold_std = np.std(tdu_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_TD_U_class_std'] = str(tdu_class_fold_std)

            cld_class_list = np.array(data_pack['Boots_CLD_class'].tolist())
            cld_class_fold_mean = np.mean(cld_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_CLD_class_mean'] = str(cld_class_fold_mean)
            cld_class_fold_std = np.std(cld_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_CLD_class_std'] = str(cld_class_fold_std)

            n1_class_list = np.array(data_pack['Boots_N1_class'].tolist())
            n1_class_fold_mean = np.mean(n1_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_N1_class_mean'] = str(n1_class_fold_mean)
            n1_class_fold_std = np.std(n1_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_N1_class_std'] = str(n1_class_fold_std)

            n2_class_list = np.array(data_pack['Boots_N2_class'].tolist())
            n2_class_fold_mean = np.mean(n2_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_N2_class_mean'] = str(n2_class_fold_mean)
            n2_class_fold_std = np.std(n2_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_N2_class_std'] = str(n2_class_fold_std)

            lsc_class_list = np.array(data_pack['Boots_LSC_class'].tolist())
            lsc_class_fold_mean = np.mean(lsc_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_LSC_class_mean'] = str(lsc_class_fold_mean)
            lsc_class_fold_std = np.std(lsc_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_LSC_class_std'] = str(lsc_class_fold_std)

            f1_class_list = np.array(data_pack['Boots_F1_class'].tolist())
            f1_class_fold_mean = np.mean(f1_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_F1_class_mean'] = str(f1_class_fold_mean)
            f1_class_fold_std = np.std(f1_class_list, axis=0).tolist()
            df_aggre.loc[condition2, 'Boots_F1_class_std'] = str(f1_class_fold_std)

    return df_aggre
