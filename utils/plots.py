import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_embedding(x, x1, labels, title=None):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    x = (x - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例

    plt.scatter(x[:, 0], x[:, 1], c=labels/len(labels), cmap="viridis")
    plt.scatter(x1[:, 0], x1[:, 1], c=labels / len(labels), cmap="viridis")
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    # plt.title(title, fontsize=14)
    # 返回值
    return fig

def t_sne(class_means, old_class_vectors, new_class_vectors, task_id, save_path):
    if old_class_vectors:
        fig = plt.figure()  # 创建图形实例
        for i in range(task_id):
            old_task_vectors = torch.cat(old_class_vectors[i*10:(i+1)*10], dim=0).detach().cpu().numpy()
            old_task_class_mean = class_means[i*10:(i+1)*10]

            X_embedded = TSNE(n_components=2, learning_rate='auto',
                              init='pca', perplexity=3).fit_transform(np.concatenate((old_task_vectors, old_task_class_mean), axis=0))
            # old_task_class_mean = TSNE(n_components=2, learning_rate='auto',
            #                   init='pca', perplexity=3).fit_transform(old_task_class_mean)
            x_min, x_max = np.min(X_embedded, axis=0), np.max(X_embedded, axis=0)
            X_embedded = (X_embedded - x_min) / (x_max - x_min)  # 对数据进行归一化处理

            old_task_vectors = X_embedded[0:-10, :]
            old_task_class_mean = X_embedded[-10:, :]

            vector_labels = np.arange(0, 5000)//500
            mean_labels = np.arange(0, 10)

            plt.subplot(2, (task_id+1)//2, i+1)
            plt.scatter(old_task_vectors[:, 0], old_task_vectors[:, 1], c=vector_labels / 10, cmap="viridis")
            plt.scatter(old_task_class_mean[:, 0], old_task_class_mean[:, 1], c=mean_labels / 10, marker="P", cmap="viridis")
            plt.xticks()  # 指定坐标的刻度
            plt.yticks()
        # fig.show()
        plt.savefig(save_path)

