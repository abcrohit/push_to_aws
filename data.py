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


# Credit data
credit_features=pd.read_csv("creditcard.csv")
credit_features.rename(columns={'Class':'class'});
# normalizing credit data
credit_features.iloc[:,:-1]=(credit_features.iloc[:,:-1]-credit_features.iloc[:,:-1].min())
credit_features.iloc[:,:-1]=credit_features.iloc[:,:-1]/credit_features.iloc[:,:-1].max()

# crypto data
data_dir="./elliptic_bitcoin_dataset/elliptic_txs_"
edges=pd.read_csv(data_dir+"edgelist.csv")
features=pd.read_csv(data_dir+"features.csv",header=None)
classes=pd.read_csv(data_dir+"classes.csv")

tx_features = ["tx_feat_"+str(i) for i in range(2,95)]
agg_features = ["agg_feat_"+str(i) for i in range(1,73)]
features.columns = ["txId","time_step"] + tx_features + agg_features
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
features['class'] = features['class'].apply(lambda x: 0 if x == "unknown" else int(x))

crypto_features=features.drop('txId',1);
# ADARSH

# normalizing crypto data
crypto_features.iloc[:,:-1]=(crypto_features.iloc[:,:-1]-crypto_features.iloc[:,:-1].min())
crypto_features.iloc[:,:-1]=crypto_features.iloc[:,:-1]/crypto_features.iloc[:,:-1].max()
# splitting between labelled and unlabelled

# labelled 
crypto_labelled=crypto_features[crypto_features['class']!=0]
crypto_labelled.iloc[:,-1]=crypto_labelled.iloc[:,-1]-2*crypto_labelled.iloc[:,-1].min()
crypto_labelled.iloc[:,-1]=crypto_labelled.iloc[:,-1]*crypto_labelled.iloc[:,-1]
# ADARSH

# 0 is legal
# 1 is illegal

# unlabelled
crypto_unlabelled=crypto_features[crypto_features['class']==0]
