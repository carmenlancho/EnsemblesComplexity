import random

import numpy as np

mylist = ["geeks", "for", "python"]
values = np.array([1,2,3,4,5,6,7,8,9,10])

print(random.choices(values, weights=[0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05], k=10))



import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

data = datasets.load_wine(as_frame = True)

X = data.data
y = data.target
weights = data.frame.index.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
weights_train = weights[y_train.index]
weights_train = np.array([(1/len(weights_train))] * len(weights_train))

estimator_range = [2,4,6,8,10,12,14,16]

models = []
scores = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train, sample_weight=weights_train)

    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

# Generate the plot of scores against number of estimators
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize plot
plt.show()




# evaluate bagging algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
weights = np.array([(1/len(y))] * len(y))

# define the model
model = BaggingClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')#,
                           # fit_params={'sample_weight': weights})
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# cross validation hand made
# kf = cross_validation.KFold(len(y), n_folds=5)
# for train_index, test_index in kf:
#
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#
#    model.fit(X_train, y_train)
#    print confusion_matrix(y_test, model.predict(X_test))


# bagging manually
# https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_bagging.html