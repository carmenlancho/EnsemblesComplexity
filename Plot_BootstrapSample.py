import matplotlib.pyplot as plt

X_bootstrap, y_bootstrap,
data = pd.DataFrame(X_bootstrap, columns=['x1','x2'])
data['y'] = y_bootstrap

data = pd.DataFrame(X_train, columns=['x1','x2'])
data['y'] = y_train
len(y_train)
len(y_bootstrap)
# Plot
# For labels
labels = list(data.index)
idx_1 = np.where(data.y == 1)
idx_0 = np.where(data.y == 0)
idx_2 = np.where(data.y == 2)
plt.scatter(data.iloc[idx_0].x1, data.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data.iloc[idx_1].x1, data.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data.iloc[idx_2].x1, data.iloc[idx_2].x2, s=30, c='k', marker="*", label='l')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()

## easy
X_bootstrap, y_bootstrap,
data_easy = pd.DataFrame(X_bootstrap, columns=['x1','x2'])
data_easy['y'] = y_bootstrap

# Plot
# For labels
labels = list(data_easy.index)
idx_1 = np.where(data_easy.y == 1)
idx_0 = np.where(data_easy.y == 0)
idx_2 = np.where(data_easy.y == 2)
plt.scatter(data_easy.iloc[idx_0].x1, data_easy.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data_easy.iloc[idx_1].x1, data_easy.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data_easy.iloc[idx_2].x1, data_easy.iloc[idx_2].x2, s=30, c='k', marker="*", label='l')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()

## hard
X_bootstrap, y_bootstrap,
data_hard = pd.DataFrame(X_bootstrap, columns=['x1','x2'])
data_hard['y'] = y_bootstrap

# Plot
# For labels
labels = list(data_hard.index)
idx_1 = np.where(data_hard.y == 1)
idx_0 = np.where(data_hard.y == 0)
idx_2 = np.where(data_hard.y == 2)
plt.scatter(data_hard.iloc[idx_0].x1, data_hard.iloc[idx_0].x2, s=30, c='C0', marker=".", label='negative')
plt.scatter(data_hard.iloc[idx_1].x1, data_hard.iloc[idx_1].x2, s=30, c='C1', marker="+", label='positive')
plt.scatter(data_hard.iloc[idx_2].x1, data_hard.iloc[idx_2].x2, s=30, c='k', marker="*", label='l')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()


