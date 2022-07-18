import torch
from torch import nn
from torch import tensor
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Function
import numpy as np
import pandas as pd
import torchmetrics
from torchmetrics import AUROC , AUC , F1Score
import matplotlib.pyplot as plt
from torchmetrics import PrecisionRecallCurve
from torch import tensor

import torchmetrics
from torchmetrics import F1Score
from torchmetrics import AUROC

def F1score(Dataset,model):
    preds=Dataset.x_train
    preds=model(preds,0)[0][:,1]
    preds=torch.exp(preds)
    target=Dataset.y_train.long()
    f1=F1Score()
    return f1(preds,target)
def Auroc(Dataset,model):
    preds=Dataset.x_train
    preds=model(preds,0)[0][:,1]
    preds=torch.exp(preds)
    target=Dataset.y_train.long()
    AU=AUROC()
    return AU(preds,target)
def Auprc(preds,target):
    pr_curve = PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(preds,target)
    Auc=AUC()
    area=Auc(recall,precision)
    return area
