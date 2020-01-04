# -*-coding utf-8 -*-
# @Time :2019/12/11 14:20
# @Author : zhouheng
# To become better
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

train_file = pd.read_csv('./data/sponge/sponge.data', header=None)
print(train_file)
print(train_file.head())
print()
def center_cluster(dataset):
    pass
def distance(vec1,vec2):
    dist=np.sqrt(np.sum(np.square(vec1 - vec2)))
    pass
def load_dataset(file):
    pass
def get_name(dataset):
    class_name=dataset.loc[:,0]
    return class_name
def trans_dataset(dataset): ## 将标签转换成数字
    attribuite=set()
    dict={}
    print(dataset.shape[1])
    for i in range(1,dataset.shape[1]):
        for j in range(len(dataset.loc[:,i].values)):
            attribuite.add(dataset.loc[:,i].values[j])
            for l in range(len(attribuite)):
                dict[str(list(attribuite)[l])]=l
            m=str(dataset.loc[:,i].values[j])
            dataset.loc[:,i].values[j]=dict[m]
    print(dataset)
trans_dataset(train_file)
def K_means():
    pass