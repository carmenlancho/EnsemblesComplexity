#### Plot hostility of the instances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

root_path = os.getcwd()

def plot_hostility(X, y, host_final_instance):
    # Dataset para pintar
    data_plot = pd.DataFrame(X, columns=['x1', 'x2'])
    data_plot['label'] = y
    # dataset_name = 'prueba'

    data_plot['complex'] = host_final_instance
    # name_pos = 'Host' + ' Pos class'
    # name_neg = 'Host' + ' Neg class'

    if (len(np.unique(y)) == 2):
        cm = plt.cm.get_cmap('viridis')
        idx_1 = np.where(data_plot['label'] == 1)
        idx_0 = np.where(data_plot['label'] == 0)

        # name_graph = str(dataset_name) + '_' + str(df_complexity.columns[i])

        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10, 4))
        images = []
        images.append([axes[0].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=80, c='gray', marker=".",
                                       label='negative'),
                       axes[0].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=80,
                                       c=data_plot.iloc[idx_1].complex,
                                       cmap=cm,
                                       marker="+", label='positive', vmin=0, vmax=1)])
        images.append([axes[1].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=80, c='gray', marker="+",
                                       label='positive'),
                       axes[1].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=80,
                                       c=data_plot.iloc[idx_0].complex,
                                       cmap=cm,
                                       marker=".", label='negative', vmin=0, vmax=1)])
        fig.colorbar(images[1][1], ax=axes, fraction=.05)
        # fig.savefig(name_graph + ".png")
        # fig.clf()
        fig.show()
    elif (len(np.unique(y)) == 3):
        cm = plt.cm.get_cmap('viridis')
        idx_2 = np.where(data_plot['label'] == 2)
        idx_1 = np.where(data_plot['label'] == 1)
        idx_0 = np.where(data_plot['label'] == 0)

        # name_graph = str(dataset_name) + '_' + str(df_complexity.columns[i])

        fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(10, 4))
        images = []
        images.append([axes[0].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[0].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[0].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30,
                                       c=data_plot.iloc[idx_1].complex,
                                       cmap=cm,
                                       marker="+", vmin=0, vmax=1)])
        images.append([axes[1].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[1].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[1].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30,
                                       c=data_plot.iloc[idx_0].complex,
                                       cmap=cm,
                                       marker=".", label='negative', vmin=0, vmax=1)])
        images.append([axes[2].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                      axes[2].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[2].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30,
                                       c=data_plot.iloc[idx_2].complex,
                                       cmap=cm,
                                       marker="v", vmin=0, vmax=1)])
        fig.colorbar(images[1][1], ax=axes, fraction=.05)
        # fig.savefig(name_graph + ".png")
        # fig.clf()
        fig.show()

    elif (len(np.unique(y)) == 5):
        cm = plt.cm.get_cmap('viridis')
        idx_4 = np.where(data_plot['label'] == 4)
        idx_3 = np.where(data_plot['label'] == 3)
        idx_2 = np.where(data_plot['label'] == 2)
        idx_1 = np.where(data_plot['label'] == 1)
        idx_0 = np.where(data_plot['label'] == 0)

        # name_graph = str(dataset_name) + '_' + str(df_complexity.columns[i])

        fig, axes = plt.subplots(nrows=1, ncols=5, constrained_layout=True, figsize=(12, 4))
        images = []
        images.append([axes[0].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[0].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[0].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[0].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[0].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30,
                                       c=data_plot.iloc[idx_1].complex,
                                       cmap=cm,
                                       marker="+", vmin=0, vmax=1)])
        images.append([axes[1].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[1].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[1].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[1].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[1].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30,
                                       c=data_plot.iloc[idx_0].complex,
                                       cmap=cm,
                                       marker=".", label='negative', vmin=0, vmax=1)])
        images.append([axes[2].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[2].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[2].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[2].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[2].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30,
                                       c=data_plot.iloc[idx_2].complex,
                                       cmap=cm,
                                       marker="v", vmin=0, vmax=1)])

        images.append([axes[3].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[3].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[3].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[3].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[3].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30,
                                       c=data_plot.iloc[idx_3].complex,
                                       cmap=cm,
                                       marker="p", vmin=0, vmax=1)])

        images.append([axes[4].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[4].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[4].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[4].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[4].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30,
                                       c=data_plot.iloc[idx_4].complex,
                                       cmap=cm,
                                       marker="s", vmin=0, vmax=1)])


        fig.colorbar(images[1][1], ax=axes, fraction=.05)
        # fig.savefig(name_graph + ".png")
        # fig.clf()
        fig.show()

    elif (len(np.unique(y)) == 6):
        cm = plt.cm.get_cmap('viridis')
        idx_5 = np.where(data_plot['label'] == 5)
        idx_4 = np.where(data_plot['label'] == 4)
        idx_3 = np.where(data_plot['label'] == 3)
        idx_2 = np.where(data_plot['label'] == 2)
        idx_1 = np.where(data_plot['label'] == 1)
        idx_0 = np.where(data_plot['label'] == 0)

        # name_graph = str(dataset_name) + '_' + str(df_complexity.columns[i])

        fig, axes = plt.subplots(nrows=1, ncols=6, constrained_layout=True, figsize=(18, 4))
        images = []
        images.append([axes[0].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[0].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[0].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[0].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[0].scatter(data_plot.iloc[idx_5].x1, data_plot.iloc[idx_5].x2, s=30, c='gray', marker="x"),
                       axes[0].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30,
                                       c=data_plot.iloc[idx_1].complex,
                                       cmap=cm,
                                       marker="+", vmin=0, vmax=1)])
        images.append([axes[1].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[1].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[1].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[1].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[1].scatter(data_plot.iloc[idx_5].x1, data_plot.iloc[idx_5].x2, s=30, c='gray', marker="x"),
                       axes[1].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30,
                                       c=data_plot.iloc[idx_0].complex,
                                       cmap=cm,
                                       marker=".", label='negative', vmin=0, vmax=1)])
        images.append([axes[2].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[2].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[2].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[2].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[2].scatter(data_plot.iloc[idx_5].x1, data_plot.iloc[idx_5].x2, s=30, c='gray', marker="x"),
                       axes[2].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30,
                                       c=data_plot.iloc[idx_2].complex,
                                       cmap=cm,
                                       marker="v", vmin=0, vmax=1)])

        images.append([axes[3].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[3].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[3].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[3].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[3].scatter(data_plot.iloc[idx_5].x1, data_plot.iloc[idx_5].x2, s=30, c='gray', marker="x"),
                       axes[3].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30,
                                       c=data_plot.iloc[idx_3].complex,
                                       cmap=cm,
                                       marker="p", vmin=0, vmax=1)])

        images.append([axes[4].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[4].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[4].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[4].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[4].scatter(data_plot.iloc[idx_5].x1, data_plot.iloc[idx_5].x2, s=30, c='gray', marker="x"),
                       axes[4].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30,
                                       c=data_plot.iloc[idx_4].complex,
                                       cmap=cm,
                                       marker="s", vmin=0, vmax=1)])

        images.append([axes[5].scatter(data_plot.iloc[idx_0].x1, data_plot.iloc[idx_0].x2, s=30, c='gray', marker="."),
                       axes[5].scatter(data_plot.iloc[idx_1].x1, data_plot.iloc[idx_1].x2, s=30, c='gray', marker="+"),
                       axes[5].scatter(data_plot.iloc[idx_2].x1, data_plot.iloc[idx_2].x2, s=30, c='gray', marker="v"),
                       axes[5].scatter(data_plot.iloc[idx_3].x1, data_plot.iloc[idx_3].x2, s=30, c='gray', marker="p"),
                       axes[5].scatter(data_plot.iloc[idx_4].x1, data_plot.iloc[idx_4].x2, s=30, c='gray', marker="s"),
                       axes[5].scatter(data_plot.iloc[idx_5].x1, data_plot.iloc[idx_5].x2, s=30,
                                       c=data_plot.iloc[idx_5].complex,
                                       cmap=cm,
                                       marker="x", vmin=0, vmax=1)])

        fig.colorbar(images[1][1], ax=axes, fraction=.05)
        # fig.savefig(name_graph + ".png")
        # fig.clf()
        fig.show()


    return




path_csv = os.chdir(root_path+'/Results_Complexity_InstanceLevel')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)

total_name_list = ['ComplexityMeasures_InstanceLevel_Dataset2.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset3.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset6.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset7.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset8.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset10.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset11.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset14.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset17.csv',
                   'ComplexityMeasures_InstanceLevel_Moon2.csv']

total_name_list = ['ComplexityMeasures_InstanceLevel_Dataset18.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset19.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset21.csv',
                   'ComplexityMeasures_InstanceLevel_Dataset22.csv']

total_name_list = ['ComplexityMeasures_InstanceLevel_Dataset23.csv']

total_name_list = ['ComplexityMeasures_InstanceLevel_Dataset20.csv']

for data_file in total_name_list:
    os.chdir(root_path + '/Results_Complexity_InstanceLevel')
    print(data_file)
    file = data_file
    complete_file = data_file[33:]
    data = pd.read_csv(file)
    host_final_instance = data['Hostility']
    host_final_instance = data['kDN']
    os.chdir(root_path + '/datasets')
    data_complete = pd.read_csv(complete_file)
    X = data_complete[['x1', 'x2']].to_numpy()
    y = data_complete[['y']].to_numpy()

    plot_hostility(X,y, host_final_instance)


