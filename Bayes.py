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

train_file = pd.read_csv('./data/Optical Recognition of Handwritten Digits/optdigits.tra', header=None)
train_target = train_file.loc[:, 64]
# print(train_file.head())

## 展示手写数字
i = 5
digits = train_file.loc[:, :63].values[i].reshape(8, 8)
title = train_file.loc[:, 64].values[i] ## 每行的最后一个是标签，代表当前的值
plt.title(title)
plt.imshow(digits, )
plt.show()
############ 结束 ################


test_file = pd.read_csv('./data/Optical Recognition of Handwritten Digits/optdigits.tes', header=None)
test_target = test_file.loc[:, 64]


# print(test_file.head())

# 随机打乱数据
def randSplit(dataset):
    l = list(dataset.index)
    random.shuffle(l)
    dataset.index = l
    dataset.index = range(dataset.shape[0])
    return dataset


# 计算准确率
def accuracy_computer(Truth, pred):
    acc = 0
    for i in range(len(Truth)):
        if Truth[i] == pred[i]:
            acc += 1
    return acc / len(Truth)


# 开始预测##
def bayes_classify(train_file, test_file):
    mean = []
    std = []
    result = []

    labels = train_file.iloc[:, -1].value_counts().index
    for i in labels:
        item = train_file.loc[train_file.iloc[:, -1] == i, :]
        m1 = item.iloc[:, :-1]
        m = item.iloc[:, :-1].mean()  # 每一列求一个平均值
        s = np.sum((item.iloc[:, :-1] - m) ** 2) / (item.shape[0])
        mean.append(m)
        std.append(s)
    means = pd.DataFrame(mean, index=labels)
    stds = pd.DataFrame(std, index=labels)

    print("正在预测...")
    since_time = time.time()  # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print('开始时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since_time)))
    for j in tqdm(range(test_file.shape[0])):
        start_time = time.time()
        jset = test_file.iloc[j, :-1].tolist()
        jprob = np.exp(-1 * (jset - means) ** 2 / (stds * 2)) / (np.sqrt(2 * np.pi * stds))  ## 正态分布公式
        jprob = jprob.fillna(1)  ## 将矩阵中所有的 nan 替换成 1
        prob = 1  # 当前实例总概率
        for k in range(test_file.shape[1] - 1):  # 遍历数据中的，每个特征
            # pp=jprob[k]
            prob *= jprob[k]
            # index =prob.values
            # a = np.argmax(prob.values)
        pred_class = prob.index[np.argmax(prob.values)]  # 得到最大概率的类别
        result.append(pred_class)

    print('结束时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          "一共消耗时间：%.2f min" % ((start_time - since_time) / 60))

    acc = accuracy_computer(test_target, result)
    print("预测结果：", result)
    print("准确率：", acc)
    return acc


if __name__ == '__main__':
    train = randSplit(train_file)
    bayes_classify(train, test_file)

plt.show()
