from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset import MyDataset


def select(x, y, low_range, high_range):
    """
    作用: 返回 x, y 中指定范围 (low_range, high_range) 中的数据
    """
    idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

    return x[idxes], y[idxes], idxes


def calculate_acc(y_pred, y_true, nb_old, increment, cal_task_acc=False):
    assert len(y_pred) == len(y_true), 'Data length error.'
    overall_acc = round((y_pred == y_true).sum()*100 / len(y_true), 2)
    known_classes = 0
    task_acc_list = []

    # Grouped accuracy
    if cal_task_acc:
        for new_classes in increment:
            idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + new_classes))[0]
            task_acc_list.append(round((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), 2))
            known_classes += new_classes
            if known_classes >= nb_old:
                break

    return overall_acc, task_acc_list


def cal_mean_class_recall(y_pred, y_true, nb_old, increment, cal_task_mcr=False):

    def task_mean_class_recall(y_pred, y_true, task_size=None):
        """ Calculate the mean class recall for the dataset X
        Note: should take task_size input seriously when calculating task-specific MCR !!!
        """
        cm = confusion_matrix(y_true, y_pred)
        right_of_class = np.diag(cm)
        num_of_class = cm.sum(axis=1)
        if task_size is None:
            task_size = cm.shape[0]
        # Can not use (right_of_class*100 / (num_of_class+1e-8)).mean() to calculate !!!
        # This is WRONG for task-specific MCR !!!
        mcr = np.around((right_of_class * 100 / (num_of_class + 1e-8)).sum() / (task_size + 1e-8), decimals=2)
        return mcr
    assert len(y_pred) == len(y_true), 'Data length error.'
    overall_mcr = task_mean_class_recall(y_pred, y_true)
    known_classes = 0
    task_mcr_list = []

    # Grouped accuracy
    if cal_task_mcr:
        for new_classes in increment:
            idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + new_classes))[0]
            task_mcr_list.append(task_mean_class_recall(y_pred[idxes], y_true[idxes], new_classes))
            known_classes += new_classes
            if known_classes >= nb_old:
                break
    return overall_mcr, task_mcr_list


def calculate_bwf(task_metric_curve, cur_task):
    """cur_task in [0, T-1]"""
    bwf = 0.
    if cur_task > 0:
        for i in range(cur_task):
            task_result = 0.
            for j in range(cur_task-i):
                task_result += task_metric_curve[i][cur_task - j] - task_metric_curve[i][i]
            bwf += task_result / (cur_task-i)
        bwf /= cur_task
    return bwf


def cal_avg_forgetting(task_metric_curve, cur_task):
    """cur_task in [0, T-1]"""
    avg_forget = 0.
    if cur_task > 0:
        avg_forget = (task_metric_curve[:cur_task, :cur_task].max(axis=1) - task_metric_curve[:cur_task, cur_task]).mean()
    return avg_forget


def extract_vectors(config, model, dataset, class_idx):
    '''select one class's feature vector'''
    class_samples, class_targets, class_data_idx = select(dataset.samples, dataset.targets, class_idx, class_idx + 1)
    class_dataset = MyDataset(class_samples, class_targets, dataset.use_path, dataset.transform)
    class_loader = DataLoader(class_dataset, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers)
    model.eval()
    all_vectors = []
    # all_logits = []
    with torch.no_grad():
        for idx, (inputs, targets, _) in enumerate(class_loader):
            output = model(inputs.cuda())
            all_vectors.append(output["features"])
            # all_logits.append(output["logits"])

    # ret = (torch.cat(all_vectors), torch.cat(all_logits))
    return torch.cat(all_vectors), class_samples, class_targets


def herding_select(vectors, m):
    selected_idx = []
    all_idxs = list(range(vectors.shape[0]))
    nomalized_vector = F.normalize(vectors, dim=1)  # 对特征向量做归一化
    class_mean = torch.mean(nomalized_vector, dim=0)

    # 防止类别数过少的情况
    if vectors.shape[0] > m:
        store_sample_size = m
    else:
        store_sample_size = vectors.shape[0]

    for k in range(1, store_sample_size + 1):
        sub_vectors = nomalized_vector[all_idxs]
        S = torch.sum(nomalized_vector[selected_idx], dim=0)
        mu_p = (sub_vectors + S) / k
        i = torch.argmin(torch.norm(class_mean - mu_p, p=2, dim=1))
        selected_idx.append(all_idxs.pop(i))

    return selected_idx


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
