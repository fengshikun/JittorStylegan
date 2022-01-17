
import os
import numpy as np
import glob
from PIL import Image
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import matplotlib.pyplot as plt
import mxnet as mx

def get_dataset(path, resolution, batch_size):
    root_path = os.path.join(path, str(resolution))
    return FolderDataset(root_path).set_attrs(batch_size=batch_size, shuffle=True, num_workers=2)





class FolderDataset(Dataset):
    def __init__(self, root_path, suffix="*.png"):
        super(FolderDataset, self).__init__()
        self.root_path = root_path
        self.use_rec = False

        self.file_lst = glob.glob(root_path + "/{}".format(suffix))
        if not len(self.file_lst):
            self.file_lst = glob.glob(root_path + "/{}".format("*.jpg"))
        # self.set_attrs(total_len=len(self.file_lst))
        # if os.path.basename(root_path) in ["8", "16", "32", "64", "128"]:
        #     image_rec_path = "{}_rec".format(root_path)
        #     path_imgrec = os.path.join(image_rec_path, 'train.rec')
        #     path_imgidx = os.path.join(image_rec_path, 'train.idx')
        #     self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        #     self.use_rec = True
        


        train_transforms = [
            transform.ToPILImage(),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        # self.transform = transform.Compose([
        # transform.ToPILImage(),
        # transform.RandomHorizontalFlip(),
        # transform.ToTensor(),
        # transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])


        self.transform = transform.Compose(train_transforms)

        self.image_array = []
        for file_path in self.file_lst:
            # image = Image.open(file_path)
            # image_copy = image.copy()
            # self.image_array.append(image_copy)
            # image.close()
            # image = plt.imread(file_path)

            image = plt.imread(file_path)
            
            
            if file_path[-4:] == '.png':
                image = image * 255
                
            image = image.astype('uint8')
            
            self.image_array.append(image)
    
    def __getitem__(self, index):
        if self.use_rec:
            s = self.imgrec.read_idx(index)
            header, img = mx.recordio.unpack(s)
            image_copy = mx.image.imdecode(img).asnumpy()
        else:
            # file_path = self.file_lst[index]
            # image = Image.open(file_path)
            # image_copy = image.copy()
            # image.close()
            image_copy = self.image_array[index]
        return self.transform(image_copy)


        image = self.image_array[index]
        return self.transform(image)
    
    def __len__(self):
        return len(self.file_lst)