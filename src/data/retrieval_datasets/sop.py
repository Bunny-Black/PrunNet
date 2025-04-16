import csv
import os


from .base_dataset import BaseDataset


class SOPDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(SOPDataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "test",'gallery','query'}

    def set_paths_and_labels(self):
        # label_fpath = os.path.join(self.data_dir, f"Ebay_{self.split}.txt")
        label_fpath = self.cfg.DATA.FILE_DIR if self.split=='train' else os.path.join(self.cfg.EVAL.ROOT, f"Ebay_{self.split}.txt")
        self.labels = []
        self.paths = []
        with open(label_fpath, "r") as f:
            reader = csv.DictReader(f, delimiter=" ")
            for row in reader:
                label = int(row["class_id"]) - 1    # 从 0 开始
                path = os.path.join(self.data_dir, row["path"])
                self.labels.append(label)
                self.paths.append(path)
        
