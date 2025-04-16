'''
新旧模型数据集标签重新映射
'''
import os 
import numpy as np

def n2o_remap_for_sop(root,old_data_list_path,new_data_list_path,n2o_remap_name = "sop_new2old_map.npy"):
    with open (os.path.join(root,old_data_list_path), 'r') as f:
        old = f.read().split('\n')[1:-1]
    with open(os.path.join(root,new_data_list_path),'r') as f1 :
        new = f1.read().split("\n")[1:-1] 
    new2old_dict = {}
    for line in old:
        line_list = line.split(" ")
        image_path = line_list[3]
        for line_new in new:
            line_new_list = line.split(" ")
            image_new_path = line_new_list[3]
            if image_path == image_new_path:
                new2old_dict.update({line_list[1]:line_new_list[1]})
    np.save(n2o_remap_name,new2old_dict)






