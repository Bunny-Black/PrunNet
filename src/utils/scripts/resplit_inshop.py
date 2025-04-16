"""
重新按比例采样inshop训练集 zikun
"""

import os
import random


def sample_train_for_insop(sample_rate, root, file_name, output_name):
    with open(os.path.join(root, file_name), 'r') as f:
        db = f.read().split("\n")[2: -1]

    fb = open(os.path.join(root, output_name), 'w')

    train_id_list, test_id_list = [], []
    # 统计训练集样本量
    for line in db:
        line_list = line.split(" ")
        line_list = list(filter(lambda x: x, line_list))
        id = int(line_list[1].split("_")[-1])
        if id not in train_id_list and line_list[-1] == "train":
            train_id_list.append(id)
        if id not in test_id_list and line_list[-1] in ["query", "gallery"]:
            test_id_list.append(id)

    num_train_ids = len(train_id_list)
    num_test_ids = len(test_id_list)
    
    # 随机采样某一比例的类别
    # sampled_train_ids = random.sample(train_id_list, int(num_train_ids * sample_rate))
    sampled_train_ids = list(range(int(num_train_ids * sample_rate)))
    print(sampled_train_ids)
    sampled_train_ids.sort()
    # 根据采样结果重写数据集划分文件
    total_samples = 0
    for line in db:
        line_list = line.split(" ")
        line_list = list(filter(lambda x: x, line_list))
        id = int(line_list[1].split("_")[-1])
        if id in sampled_train_ids and line_list[-1] == "train":
            fb.writelines(line)
            fb.write('\n')
            total_samples += 1
        if line_list[-1] in ["query", "gallery"]:
            fb.writelines(line)
            fb.write('\n')
            total_samples += 1
    fb.close()
    return total_samples

def remap_class(root, file_name, output_name, total_samples):
    with open(os.path.join(root, file_name), 'r') as f:
        db = f.read().split("\n")[: -1]

    fb = open(os.path.join(root, output_name), 'w')
    fb.writelines(str(total_samples))
    fb.write('\n')
    fb.writelines("image_name item_id evaluation_status")
    fb.write('\n')
    labels = []
    classes = -1
    for line in db:
        line = line.split(" ")
        line = list(filter(lambda x: x, line))
        num = int(line[1].split("_")[-1])
        if len(labels) == 0 or num not in labels:
            labels.append(num)
            classes=classes+1
            line[1] = "id_"+str(classes).zfill(8)
            fb.writelines(line[0].ljust(80)+"   "+line[1]+"  "+line[2])
            fb.write('\n')

        else:
            line[1] = "id_" + str(classes).zfill(8)
            fb.writelines(line[0].ljust(80) + "   " + line[1] + "  " + line[2])
            fb.write('\n')
    fb.close()



if __name__ == "__main__":
    random.seed(42)
    
    for i in range(1, 11):
        sample_rate = i / 10.0
        total_samples = sample_train_for_insop(sample_rate=sample_rate, 
                           root="/he_zy/zhaokai/data/In-shop_Clothes_Retrieval_Benchmark/Img", 
                           file_name="list_eval_partition.txt", 
                           output_name="ordered_list_eval_partition_{:.2f}.txt".format(sample_rate))
        """
        remap_class(root="/he_zy/zhaokai/data/In-shop_Clothes_Retrieval_Benchmark/Img", 
                file_name="list_eval_partition_{:.2f}.txt".format(sample_rate),
                output_name="list_eval_partition_{:.2f}_remap.txt".format(sample_rate), 
                total_samples = total_samples)
        """
    #sample_rate = 0.05
    #total_samples = sample_train_for_insop(sample_rate=sample_rate, 
    #                       root="/he_zy/zhaokai/data/In-shop_Clothes_Retrieval_Benchmark/Img", 
    #                       file_name="list_eval_partition.txt", 
    #                       output_name="list_eval_partition_{:.2f}.txt".format(sample_rate))
    #remap_class(root="/he_zy/zhaokai/data/In-shop_Clothes_Retrieval_Benchmark/Img", 
    #            file_name="list_eval_partition_{:.2f}.txt".format(sample_rate),
    #            output_name="list_eval_partition_{:.2f}_remap.txt".format(sample_rate), 
    #            total_samples = total_samples)
    
    # SOP这个数据集、new2old map