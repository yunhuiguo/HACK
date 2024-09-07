import torch
from torchvision import datasets, transforms
import torchvision

import numpy as np
import os.path as osp
from PIL import Image
import os,  re
from collections import defaultdict


from typing import List, Tuple

class MSBaseDataSet(torch.utils.data.Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """
    def __init__(self, conf, img_source, transform=None, mode="RGB"):
        self.mode = mode

        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

        self.num_cls = len(self.label_index_dict.keys())
        self.num_train = len(self.label_list)

        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def read_image(self, img_path, mode='RGB'):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError(f"{img_path} does not exist")
        while not got_img:
            try:
                img = Image.open(img_path).convert("RGB")
                if mode == "BGR":
                    r, g, b = img.split()
                    img = Image.merge("RGB", (b, g, r))
                got_img = True
            except IOError:
                print(f"IOError incurred when reading '{img_path}'. Will redo.")
                pass
        return img

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = self.read_image(img_path, mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, int(label)



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class ThreeCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, transform_val):
        self.transform = transform
        self.transform_val = transform_val

    def __call__(self, x):
        return [self.transform_val(x), self.transform(x), self.transform(x)]



class IMBALANCEMNIST_plain(datasets.MNIST):
    cls_num = 10

    def __init__(self, root, class_idx=0, sample_per_class=100, imb_type='exp', imb_factor=1.0, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        
        super(IMBALANCEMNIST_plain, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)

        if train:

            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)

            img_num_list = []

            for i in range(10):
                
                if i == class_idx:
                    img_num_list.append(10000)
            
                else:
                    img_num_list.append(0)

            self.gen_imbalanced_data(img_num_list)



    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        
        #img_max = len(self.data) / cls_num
        img_max = 0
        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)
        

        idx = np.where(targets_np == 0)[0]
        img_max = max(img_max, idx.shape[0])

        img_num_per_cls = [] 

        if imb_factor != 1.0:
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
           
            elif imb_type == 'step':
            
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max * imb_factor))
            else:
                img_num_per_cls.extend([int(img_max)] * cls_num)
        else:
            for the_class in classes:
                idx = np.where(targets_np == the_class)[0]
                img_num_per_cls.append(idx.shape[0])

        return img_num_per_cls



    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)

        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):

            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]

            np.random.shuffle(idx)

            selec_idx = idx[:the_img_num]

            new_data.append(self.data[selec_idx])

            new_targets.extend([the_class, ] * the_img_num)
            
        new_data = torch.tensor(np.vstack(new_data))
       
        self.data = new_data
        self.targets = new_targets



    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


#########################################
class CongealingMNIST(datasets.MNIST):

    def __init__(self, args=None, root="/home/yunhui/hyperbolic_assignment/datasets/congealing_mnist/", hyperbolic_embeddings=None, rand_number=0, train=True,
             transform=None, target_transform=None,
             download=False):

        super(CongealingMNIST, self).__init__(root, train, transform, target_transform, download)

        np.random.seed(rand_number)

        congealing_Imgs = np.load(root + "congealing_Imgs.npy")

        original_Imgs = np.load(root + "original_Imgs.npy")


        w, h, count = congealing_Imgs.shape

        congealing_Imgs = np.moveaxis(congealing_Imgs, -1,  0)[:500, :]*255
        
        original_Imgs   = np.moveaxis(original_Imgs,   -1,  0)[500:, :]*255


        self.data = torch.tensor(np.concatenate((congealing_Imgs, original_Imgs), axis=0)).to(torch.uint8)

        self.congealing_index = torch.cat((torch.tensor(np.ones(500)), torch.tensor(np.zeros(count-500))))



        self.train_nat = hyperbolic_embeddings
            
        self.targets = torch.tensor(np.ones(count))




    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        """
        Helper method that update the assigned feature representation
        Used after cost minimisation every few epoch.
        :param indexes:
        :param new_targets:
        :return:
        """
        if self.train:

            self.train_nat[indexes, :] = new_targets
        
        else:
        
            self.test_nat[indexes, :] = new_targets


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        x, y = super().__getitem__(index)


        if not self.train:
            return index, x, y, self.test_nat[index, :], self.congealing_index[index]
        else:
            return index, x, y, self.train_nat[index, :], self.congealing_index[index]

#########################################



class IMBALANCEMNIST(datasets.MNIST):
    cls_num = 10

    def __init__(self, args, root, class_idx, hyperbolic_embeddings, sample_per_class=100, imb_type='exp', imb_factor=1.0, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        
        np.random.seed(rand_number)

        self.train_nat = hyperbolic_embeddings


        if train:

            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            
            img_num_list = []


            if args.all_classes:

                for i in range(10):
                
                    img_num_list.append(1000)
            else:

                for i in range(10):
                
                    if i == class_idx:
                        img_num_list.append(10000)
                
                    else:
                        img_num_list.append(0)

            self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        
        #img_max = len(self.data) / cls_num
        img_max = 0
        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)
        

        idx = np.where(targets_np == 0)[0]
        img_max = max(img_max, idx.shape[0])

        img_num_per_cls = [] 

        if imb_factor != 1.0:
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
           
            elif imb_type == 'step':
            
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max * imb_factor))
            else:
                img_num_per_cls.extend([int(img_max)] * cls_num)
        else:
            for the_class in classes:
                idx = np.where(targets_np == the_class)[0]
                img_num_per_cls.append(idx.shape[0])

        return img_num_per_cls


    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)

        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):

            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]

            np.random.shuffle(idx)

            selec_idx = idx[:the_img_num]

            new_data.append(self.data[selec_idx])

            new_targets.extend([the_class, ] * the_img_num)
            
        new_data = torch.tensor(np.vstack(new_data))
       
        self.data = new_data
        self.targets = new_targets



    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        """
        Helper method that update the assigned feature representation
        Used after cost minimisation every few epoch.
        :param indexes:
        :param new_targets:
        :return:
        """
        if self.train:

            self.train_nat[indexes, :] = new_targets
        
        else:
        
            self.test_nat[indexes, :] = new_targets


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        x, y = super().__getitem__(index)


        if not self.train:
            return index, x, y, self.test_nat[index, :]
        else:
            return index, x, y, self.train_nat[index, :]



class IMBALANCEMNIST_tree(datasets.MNIST):
    cls_num = 10

    def __init__(self, args, root, class_idx, tree_embeddings, sample_per_class=100, imb_type='exp', imb_factor=1.0, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCEMNIST_tree, self).__init__(root, train, transform, target_transform, download)

        np.random.seed(rand_number)

        self.train_nat = tree_embeddings


        print (self.train_nat)

        if train:

            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            
            img_num_list = []

            for i in range(10):
            
                if i == class_idx:
                    img_num_list.append(sample_per_class)
            
                else:
                    img_num_list.append(0)

            self.gen_imbalanced_data(img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        
        #img_max = len(self.data) / cls_num
        img_max = 0
        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)
        

        idx = np.where(targets_np == 0)[0]
        img_max = max(img_max, idx.shape[0])

        img_num_per_cls = [] 

        if imb_factor != 1.0:
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
           
            elif imb_type == 'step':
            
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max * imb_factor))
            else:
                img_num_per_cls.extend([int(img_max)] * cls_num)
        else:
            for the_class in classes:
                idx = np.where(targets_np == the_class)[0]
                img_num_per_cls.append(idx.shape[0])

        return img_num_per_cls


    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)

        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):

            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]

            np.random.shuffle(idx)

            selec_idx = idx[:the_img_num]

            new_data.append(self.data[selec_idx])

            new_targets.extend([the_class, ] * the_img_num)
            
        new_data = torch.tensor(np.vstack(new_data))
       
        self.data = new_data
        self.targets = new_targets



    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        """
        Helper method that update the assigned feature representation
        Used after cost minimisation every few epoch.
        :param indexes:
        :param new_targets:
        :return:
        """
        if self.train:

            self.train_nat[indexes, :] = new_targets
        
        else:
        
            self.test_nat[indexes, :] = new_targets


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = super().__getitem__(index)


        if not self.train:
            return index, x, y, self.test_nat[index, :]
        else:
            return index, x, y, self.train_nat[index, :]





class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, args, root, class_idx,  hyperbolic_embeddings, imb_type='exp', imb_factor=1.0, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        
        old_img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        
        self.train_nat = hyperbolic_embeddings

        img_num_list = []

        for i in range(self.cls_num):
            
            if i == class_idx:

                img_num_list.append(old_img_num_list[i])
        
            else:
                img_num_list.append(0)

        self.gen_imbalanced_data(img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls



    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        """
        Helper method that update the assigned feature representation
        Used after cost minimisation every few epoch.
        :param indexes:
        :param new_targets:
        :return:
        """
        if self.train:

            self.train_nat[indexes, :] = new_targets
        
        else:
        
            self.test_nat[indexes, :] = new_targets


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        x, y = super().__getitem__(index)

        if not self.train:
            return index, x, y, self.test_nat[index, :]
        else:
            return index, x, y, self.train_nat[index, :]




class IMBALANCECIFAR10_plain(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root,  class_idx, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10_plain, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        old_img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        

        img_num_list = []

        for i in range(self.cls_num):
            
            if i == class_idx:

                img_num_list.append(old_img_num_list[i])
        
            else:
                img_num_list.append(0)

        self.gen_imbalanced_data(img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls



    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list




class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


