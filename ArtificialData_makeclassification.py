# Generation of artificial data with make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import os
root_path = os.getcwd()

X, y = make_classification(
    n_samples=500, # 3000 observations
    n_features=2, # 2 total features
    n_informative=2,n_redundant=0,
    n_classes=2, # binary target/label
    n_clusters_per_class = 2,
    hypercube = True,
    shift = 0.0, # just changes the scale
    shuffle = True, # Shuffle the samples and the features
    class_sep = 0.2,
    flip_y=0.4,
    random_state=14 #
)

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y

# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.show()




os.chdir(root_path + '/datasets')
data.to_csv('data1_makeclass.csv', index=False)

# X, y = make_classification(
#     n_samples=3000, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 2,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.9,
#     flip_y=0.04,
#     random_state=999 #
# )


data.to_csv('data2_makeclass.csv', index=False)
# X, y = make_classification(
#     n_samples=3000, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 2,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.3,
#     flip_y=0.2,
#     random_state=4 #
# )


data.to_csv('data3_makeclass.csv', index=False)
# X, y = make_classification(
#     n_samples=3000, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 2,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.3,
#     flip_y=0.0,
#     random_state=21 #
# )


data.to_csv('data4_makeclass.csv', index=False)
# X, y = make_classification(
#     n_samples=3000, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 2,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.6,
#     flip_y=0.0,
#     random_state=28 #
# )

data.to_csv('Dataset14.csv', index=False)
# X, y = make_classification(
#     n_samples=1000, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 1,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.5,
#     flip_y=0.4,
#     random_state=3 #
# )


data.to_csv('Dataset15.csv', index=False)
# X, y = make_classification(
#     n_samples=2000, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 1,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.3,
#     flip_y=0.4,
#     random_state=6 #
# )

data.to_csv('Dataset16.csv', index=False)
# X, y = make_classification(
#     n_samples=500, # 3000 observations
#     n_features=2, # 2 total features
#     n_informative=2,n_redundant=0,
#     n_classes=2, # binary target/label
#     n_clusters_per_class = 2,
#     hypercube = True,
#     shift = 0.0, # just changes the scale
#     shuffle = True, # Shuffle the samples and the features
#     class_sep = 0.2,
#     flip_y=0.4,
#     random_state=14 #
# )

from sklearn.datasets import make_blobs, make_moons, make_checkerboard

X,y = make_moons(n_samples=3000, shuffle=True, noise=0.33, random_state=28)

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y

# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.show()


data.to_csv('data5_makemoons.csv', index=False)
# X,y = make_moons(n_samples=3000, shuffle=True, noise=0.33, random_state=28)


