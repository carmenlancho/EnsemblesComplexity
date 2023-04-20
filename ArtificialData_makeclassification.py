# Generation of artificial data with make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs, make_moons
import os
root_path = os.getcwd()

X, y = make_classification(
    n_samples=1000, # 3000 observations
    n_features=2, # 2 total features
    n_informative=2,n_redundant=0,
    n_classes=3, # binary target/label
    n_clusters_per_class = 1,
    hypercube = True,
    shift = 0.0, # just changes the scale
    shuffle = True, # Shuffle the samples and the features
    class_sep = 1,
    flip_y=0.0,
    random_state=1 #
)

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y

# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
idx_2 = np.where(data.y == 2)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=30, c='C2', marker="+", label='l')
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



from SymbolicGeneration import gen_classification_symbolic

# data_aux = gen_classification_symbolic(m='((x1^2)/3-(x2^2)/15)',n_samples=500,flip_y=0.01)
# data=pd.DataFrame(data_aux, columns=['x'+str(i) for i in range(1,3)]+['y'])

data_aux = gen_classification_symbolic(m='x1-3*sin(x2/2)',n_samples=2000,flip_y=0)
data=pd.DataFrame(data_aux, columns=['x'+str(i) for i in range(1,3)]+['y'])

# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.show()

data.to_csv('Dataset17.csv', index=False)






def normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X0 = mn0.rvs(size=n0, random_state=seed0)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X1 = mn1.rvs(size=n1, random_state=seed1)

    mn2 = multivariate_normal(mean=mu2, cov=sigma2)
    X2 = mn2.rvs(size=n2, random_state=seed2)

    X = np.vstack((X0, X1, X2))
    y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2))

    data = pd.DataFrame(X, columns=['x', 'y'])

    # Plot
    # For labels
    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    idx_2 = np.where(y == 2)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=70, c='C0', marker=".", label='0')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=70, c='C1', marker="+", label='1')
    plt.scatter(data.iloc[idx_2].x, data.iloc[idx_2].y, s=70, c='k', marker="*", label='2')
    # for i, txt in enumerate(labels):
    #    plt.annotate(txt, (data.iloc[i, 0], data.iloc[i, 1]))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    return X, y



## dataset 18
seed0 = 1
seed1 = 2
seed2 = 4
n0 = 500
n1 = 500
n2 = 500

mu0 = [0.5, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [2.5, -2.5]
sigma1 = [[1, 0], [0, 1]]
mu2 = [-1, -3]
sigma2 = [[1, 0], [0, 1]]
X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y
data.to_csv('Dataset18.csv', index=False)


## dataset 19
seed0 = 1
seed1 = 2
seed2 = 3
n0 = 500
n1 = 500
n2 = 500

mu0 = [0.5, 0]
sigma0 = [[7, 0], [0, 1]]
mu1 = [-2, -6]
sigma1 = [[1, -2], [-2, 7]]
mu2 = [4, -5]
sigma2 = [[2, 3], [3,7]]
X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)
data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y
data.to_csv('Dataset19.csv', index=False)


## dataset 20
seed0 = 1
seed1 = 2
seed2 = 3
seed3 = 4
seed4 = 5
seed5 = 6
n0 = 500; n1 = 500; n2 = 500; n3 = 500; n4 = 500; n5 = 500
mu0 = [0, 0]
sigma0 = [[2, 0], [0, 1]]
mu1 = [-2, -4]
sigma1 = [[1, -1], [-1, 3]]
mu2 = [3, -4]
sigma2 = [[1, 1], [1,3]]

mu3 = [-3, 0]
sigma3 = [[0.5, 0.2], [0.2,1]]
mu4 = [0.5, -7.5]
sigma4 = [[0.6, 0], [0,0.6]]
mu5 = [3, 0]
sigma5 = [[0.5, 0.2], [0.2,0.5]]

mn0 = multivariate_normal(mean=mu0, cov=sigma0)
X0 = mn0.rvs(size=n0, random_state=seed0)
mn1 = multivariate_normal(mean=mu1, cov=sigma1)
X1 = mn1.rvs(size=n1, random_state=seed1)
mn2 = multivariate_normal(mean=mu2, cov=sigma2)
X2 = mn2.rvs(size=n2, random_state=seed2)

mn3 = multivariate_normal(mean=mu3, cov=sigma3)
X3 = mn3.rvs(size=n3, random_state=seed3)
mn4 = multivariate_normal(mean=mu4, cov=sigma4)
X4 = mn4.rvs(size=n4, random_state=seed4)
mn5 = multivariate_normal(mean=mu5, cov=sigma5)
X5 = mn5.rvs(size=n5, random_state=seed5)


X = np.vstack((X0, X1, X2,X3,X4,X5))
y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2)+ [3] * len(X3)+ [4] * len(X4)+ [5] * len(X5))

data = pd.DataFrame(X, columns=['x', 'y'])

# Plot
idx_1 = np.where(y == 1)
idx_0 = np.where(y == 0)
idx_2 = np.where(y == 2)
idx_3 = np.where(y == 3)
idx_4 = np.where(y == 4)
idx_5 = np.where(y == 5)
plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=30, c='C0', marker=".", label='0')
plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=30, c='C1', marker="+", label='1')
plt.scatter(data.iloc[idx_2].x, data.iloc[idx_2].y, s=30, c='k', marker="*", label='2')
plt.scatter(data.iloc[idx_3].x, data.iloc[idx_3].y, s=10, c='gold', marker="v", label='3')
plt.scatter(data.iloc[idx_4].x, data.iloc[idx_4].y, s=30, c='green', marker="1", label='4')
plt.scatter(data.iloc[idx_5].x, data.iloc[idx_5].y, s=10, c='pink', marker="h", label='5')
plt.show()

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y
data.to_csv('Dataset20.csv', index=False)



## Dataset 21
# moon + two normals

X,y = make_moons(n_samples=1000, shuffle=True, noise=0.10, random_state=28)

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

# Nos quedamos solo con la clase 0
df = data[data["y"] == 0]



seed0 = 1
seed1 = 2
n0 = 500; n1 = 500
mu0 = [0, -0.5]
sigma0 = [[0.1, 0], [0, 0.1]]
mu1 = [1.3, -1]
sigma1 = [[1, 0], [0, 1]]


mn0 = multivariate_normal(mean=mu0, cov=sigma0)
X0 = mn0.rvs(size=n0, random_state=seed0)
mn1 = multivariate_normal(mean=mu1, cov=sigma1)
X1 = mn1.rvs(size=n1, random_state=seed1)


X = np.vstack((X0, X1))
y = np.array([1] * len(X0) + [2] * len(X1))

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y
data = pd.concat([df,data])

labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
idx_2 = np.where(data.y == 2)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='0')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='1')
plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=30, c='k', marker="*", label='2')
plt.show()

# data = pd.DataFrame(X, columns=['x1','x2'])
# data['y'] = y
data.to_csv('Dataset21.csv', index=False)


## Dataset 22
# symbolic + triangle

from SymbolicGeneration import gen_classification_symbolic

# data_aux = gen_classification_symbolic(m='((x1^2)/3-(x2^2)/15)',n_samples=500,flip_y=0.01)
# data=pd.DataFrame(data_aux, columns=['x'+str(i) for i in range(1,3)]+['y'])

data_aux = gen_classification_symbolic(m='x1-3*sin(x2/2)',n_samples=2000,flip_y=0)
data=pd.DataFrame(data_aux, columns=['x'+str(i) for i in range(1,3)]+['y'])

# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.show()



# Triangle

import math
import random

# import matplotlib.pyplot as plt

def trisample(A, B, C):
    """
    Given three vertices A, B, C,
    sample point uniformly in the triangle
    """
    r1 = random.random()
    r2 = random.random()

    s1 = math.sqrt(r1)

    x = A[0] * (1.0 - s1) + B[0] * (1.0 - r2) * s1 + C[0] * r2 * s1
    y = A[1] * (1.0 - s1) + B[1] * (1.0 - r2) * s1 + C[1] * r2 * s1

    return (x, y)

random.seed(1)
A = (5, -3)
B = (11, 14)
C = (16, 11)
points = [trisample(A, B, C) for _ in range(1000)]

xx, yy = zip(*points)
plt.scatter(xx, yy, s=0.2)
plt.show()

df2 = pd.DataFrame(points,columns=['x1','x2'])
df2['y'] = 2

data = pd.concat([data,df2])

labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
idx_2 = np.where(data.y == 2)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='0')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='1')
plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=10, c='k', marker="*", label='2')
plt.show()

data.to_csv('Dataset22.csv', index=False)
## Triangle
# https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain

# data2 = np.random.triangular(-3, 0, 8, 100000)
# df2 = pd.DataFrame(data2,columns=['x1','x2'])
# df2['y'] = 2
#
# N = 1000 # number of points to create in one go
#
# rvs = np.random.random((N, 2)) # uniform on the unit square
# # Now use the fact that the unit square is tiled by the two triangles
# # 0 <= y <= x <= 1 and 0 <= x < y <= 1
# # which are mapped onto each other (except for the diagonal which has
# # probability 0) by swapping x and y.
# # We use this map to send all points of the square to the same of the
# # two triangles. Because the map preserves areas this will yield
# # uniformly distributed points.
# rvs = np.where(rvs[:, 0, None]>rvs[:, 1, None], rvs, rvs[:, ::-1])
#
#
# xmin, ymin, xmax, ymax = -0.1, 1.1, 2.0, 3.3
# rvs = np.array((ymin, xmin)) + rvs*(ymax-ymin, xmax-xmin)
# df2 = pd.DataFrame(rvs,columns=['x1','x2'])
#
# plt.scatter(df2.x1, df2.x2, s=30, c='C1', marker="+", label='positive')
# plt.show()


#### Dataset 23: mix of shapes

X,y = make_moons(n_samples=1000, shuffle=True, noise=0.10, random_state=28)

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y

# Plot
# For labels
# labels = list(data.index)
# idx_1 = np.where(data.y == 1)
# idx_0 = np.where(data.y == 0)
# plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
# plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
# plt.show()

# Nos quedamos solo con la clase 0
df = data[data["y"] == 1]
df = df*3



seed0 = 1
seed1 = 2
n0 = 500; n1 = 500
mu0 = [-1, 2]
sigma0 = [[1, 2], [2, 5]]
mu1 = [3, -3]
sigma1 = [[6, 0], [0, 6]]


mn0 = multivariate_normal(mean=mu0, cov=sigma0)
X0 = mn0.rvs(size=n0, random_state=seed0)
mn1 = multivariate_normal(mean=mu1, cov=sigma1)
X1 = mn1.rvs(size=n1, random_state=seed1)


X = np.vstack((X0, X1))
y = np.array([0] * len(X0) + [1] * len(X1))

data = pd.DataFrame(X, columns=['x1','x2'])
data['y'] = y
data = pd.concat([df,data])

# labels = list(data.index)
# idx_1 = np.where(data.y == 1)
# idx_0 = np.where(data.y == 0)
# idx_2 = np.where(data.y == 3)
# plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='0')
# plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='1')
# plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=30, c='k', marker="*", label='2')
# plt.show()




## square
min_d0 = -6
min_d1 = -4
max_d0 = -1
max_d1 = -1
data2 = np.random.uniform((min_d0, min_d1), (max_d0, max_d1), (500, 2))
df2 = pd.DataFrame(data2,columns=['x1','x2'])
df2['y'] = 4

# plt.scatter(df2.x1, df2.x2, s=30, c='C1', marker="+", label='positive')
# plt.show()

data = pd.concat([data,df2])


random.seed(1)
A = (3, -1)
B = (2, 10)
C = (5, 6)
points = [trisample(A, B, C) for _ in range(500)]

xx, yy = zip(*points)
plt.scatter(xx, yy, s=0.2)
plt.show()

df3 = pd.DataFrame(points,columns=['x1','x2'])
df3['y'] = 2

data = pd.concat([data,df3])




labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
idx_2 = np.where(data.y == 3)
idx_3 = np.where(data.y == 4)
idx_4 = np.where(data.y == 2)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='0')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='1')
plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=30, c='k', marker="*", label='2')
plt.scatter(data.iloc[idx_3].x1, data.iloc[idx_3].x2, s=30, c='gold', marker="v", label='3')
plt.scatter(data.iloc[idx_4].x1, data.iloc[idx_4].x2, s=30, c='green', marker="p", label='4')
plt.show()


# data = pd.DataFrame(X, columns=['x1','x2'])
# data['y'] = y
data.to_csv('Dataset23.csv', index=False)











