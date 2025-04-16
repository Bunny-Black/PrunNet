'''
统计inshop数据集训练集，测试集类别个数 赵凯
'''
import os

with open("./split_files/list_eval_partition.txt") as f:
    db = f.read().split("\n")[2:-1]

paths = []
query_labels = []
train_labels = []
gallery_labels = []
for line in db:
    line = line.split(" ")
    line = list(filter(lambda x: x, line))
    if line[2] == "query":
        query_labels.append(int(line[1].split("_")[-1]))
    elif line[2] == "train":
        train_labels.append(int(line[1].split("_")[-1]))
    else:
        gallery_labels.append(int(line[1].split("_")[-1]))



query_labels = list(set(query_labels))
train_labels = list(set(train_labels))
gallery_labels = list(set(gallery_labels))
print(train_labels)
print(len(query_labels))
print(len(train_labels))
print(len(gallery_labels))
