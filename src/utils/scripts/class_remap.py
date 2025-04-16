'''
inshop数据集类别重新映射 赵凯
'''
import os

with open("list_eval_partition.txt") as f:
    db = f.read().split("\n")[2:-1]

if os.path.exists("dest.txt"):
    os.remove("dest.txt")
fb = open("dest.txt", 'w')



paths = []
labels = []
train_lables = []
gallery_labels = []
classes = -1
for line in db:
    line = line.split(" ")
    line = list(filter(lambda x: x, line))
    num = int(line[1].split("_")[-1])
    if len(labels) == 0 or num not in labels:
        labels.append(num)
        classes=classes+1
        line[1] = "id_"+str(classes).zfill(8)
        fb.writelines(line[0].ljust(70)+"   "+line[1]+"  "+line[2])
        fb.write('\n')

    else:
        line[1] = "id_" + str(classes).zfill(8)
        fb.writelines(line[0].ljust(70) + "   " + line[1] + "  " + line[2])
        fb.write('\n')

fb.close()