###########################################################################################################
####  In this script we carry out an EDA of the datasets from the point of view of performance
#### and complexity in order to determine the characteristics of those datasets that benefit more
#### from the application of ensemble methos
###########################################################################################################

# We would like to answer the following questions:
# When to use ensemble methods?
# Do harder datasets benefit more from ensemble methods?
# Which datasets obtain better accuracy with ensemble rather than single classifiers?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import pyplot
import seaborn as sns
from sklearn import tree
from sklearn.tree import export_text


root_path = os.getcwd()

path_csv = os.chdir(root_path+'/Results_general_algorithm')
df_complexity = pd.read_csv('complex_info_total.csv')
path_csv = os.chdir(root_path+'/Classification_SingleLearner')
df_classif = pd.read_csv('ClassificationSingleLearner_AllDatasets.csv')
df_bagg = pd.read_csv('best_n_trees_df.csv')
df_bagg.drop(['best_n_ensemble'], axis=1, inplace=True)
df_bagg.columns = ['name_data','acc_bagging']



# We keep only the 9 selected complexity measures: Hostility, kDN, DCP, TD_U, CLD, N1, N2, LSC, F1
df_complexity.drop(['DS','TD_P','MV','CB','LSradius','H','U',
                    'F2','F3','F4'], axis=1, inplace=True)

# Join of two tables
df = pd.merge(df_complexity, df_classif, left_on='Dataset', right_on='dataset', how='left').drop('dataset', axis=1)
df = pd.merge(df, df_bagg, left_on='Dataset', right_on='name_data', how='left').drop('name_data', axis=1)


### Creation of dummy variable to indicate which method obtains higher accuracy (Bagging or Single Classifier)
df['max_acc_single_class'] = df[['acc_svmlinear', 'acc_svmrbf', 'acc_mlp', 'acc_knn', 'acc_dt',
       'acc_nb']].values.max(axis=1)
df['bagging_wins'] = 0
df.loc[(df['acc_bagging'] > df['max_acc_single_class']),'bagging_wins'] = 1
df['max_acc'] = df[['max_acc_single_class', 'acc_bagging']].values.max(axis=1)
df['diff_acc_bagg_with_single'] = df['acc_bagging'] - df['max_acc_single_class']


## Summary accuracy
df.groupby(['bagging_wins']).agg({'max_acc': [np.size,np.mean, np.median, np.std]})

summ_complexity = df.groupby(['bagging_wins']).agg({'Hostility': [np.mean, np.median, np.std],
                                  'kDN': [np.mean, np.median, np.std],
                                  'DCP': [np.mean, np.median, np.std],
                                  'TD_U': [np.mean, np.median, np.std],
                                  'CLD': [np.mean, np.median, np.std],
                                  'N1': [np.mean, np.median, np.std],
                                  'N2': [np.mean, np.median, np.std],
                                  'LSC': [np.mean, np.median, np.std],
                                  'F1': [np.mean, np.median, np.std]})




pyplot.hist(df.loc[df['bagging_wins']==1,'Hostility'], alpha=0.5, label='Bagging wins')
pyplot.hist(df.loc[df['bagging_wins']==0,'Hostility'], alpha=0.5, label='Bagging loses')
pyplot.legend(loc='upper right')
pyplot.title('Hostility')
plt.show()

#create scatterplot with regression line
sns.regplot(x=df['Hostility'],y=df['diff_acc_bagg_with_single'])
plt.hlines(y=0,xmin=0,xmax=max(df['Hostility']), colors = 'crimson')
plt.xlabel('Hostility')
plt.ylabel('Acc. Bagging - Acc. Single classifier (max)')
plt.show()

## Explicative decision tree

X = df[['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC','F1']]
Y = df['bagging_wins']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
tree.plot_tree(clf)
plt.show()

CM_names = X.columns

plt.figure(figsize=(30,20), facecolor ='white')
tree.plot_tree(clf,
               rounded=True,
               filled = True,
               feature_names = CM_names,
               class_names=True)
plt.show()


tree_rules = export_text(clf,feature_names = list(CM_names))
print(tree_rules)
