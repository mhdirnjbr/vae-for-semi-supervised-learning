import errno
import os
import PIL
import csv
from functools import reduce
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from typing import Any, Callable, List, Optional, Tuple, Union
from collections import namedtuple
import shutil

CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA(VisionDataset):
    base_folder = "celeba"
    # There currently does not appear to be an easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))
    
    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        # Mettez à jour le chemin vers vos fichiers locaux
        local_data_folder = "celeba"

        # Copiez les fichiers locaux vers le dossier de destination
        for (_, _, filename) in self.file_list:
            src_path = os.path.join(local_data_folder, filename)
            dest_path = os.path.join(self.root, self.base_folder, filename)
            shutil.copy(src_path, dest_path)

        # Extrait l'archive localement
        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

def split_celeba(X, y, sup_frac, validation_num):
    """
    splits celeba
    """

    # validation set is the last 10,000 examples
    X_valid = X[-validation_num:]
    y_valid = y[-validation_num:]

    X = X[0:-validation_num]
    y = y[0:-validation_num]

    if sup_frac == 0.0:
        return None, None, X, y, X_valid, y_valid

    if sup_frac == 1.0:
        return X, y, None, None, X_valid, y_valid

    split = int(sup_frac * len(X))
    X_sup = X[0:split]
    y_sup = y[0:split]
    X_unsup = X[split:]
    y_unsup = y[split:]

    return X_sup, y_sup, X_unsup, y_unsup, X_valid, y_valid


CELEBA_LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows', \
                 'Chubby', 'Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes', 'No_Beard', 'Oval_Face', \
                 'Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', \
                'Wearing_Necklace', 'Wearing_Necktie', 'Young']

CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair','Brown_Hair','Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', \
                      'No_Beard', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']


class CELEBACached(CelebA):
    """
    a wrapper around CelebA to load and cache the transformed data
    once at the beginning of the inference
    """
    # static class variables for caching training data
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    train_data, test_labels = None, None
    prior = torch.ones(1, len(CELEBA_EASY_LABELS)) / 2
    fixed_imgs = None
    validation_size = 20000
    data_valid, labels_valid = None, None

    def prior_fn(self):
        return CELEBACached.prior

    def clear_cache():
        CELEBACached.train_data, CELEBACached.test_labels = None, None

    def __init__(self, mode, sup_frac=None, *args, **kwargs):
        super(CELEBACached, self).__init__(split='train' if mode in ["sup", "unsup", "valid"] else 'test', *args, **kwargs)
        self.sub_label_inds = [i for i in range(len(CELEBA_LABELS)) if CELEBA_LABELS[i] in CELEBA_EASY_LABELS]
        self.mode = mode
        self.transform = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor()
                            ])

        assert mode in ["sup", "unsup", "test", "valid"], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            if CELEBACached.train_data is None:
                print("Splitting Dataset")

                CELEBACached.train_data = self.filename
                CELEBACached.train_targets = self.attr

                CELEBACached.train_data_sup, CELEBACached.train_labels_sup, \
                    CELEBACached.train_data_unsup, CELEBACached.train_labels_unsup, \
                    CELEBACached.data_valid, CELEBACached.labels_valid = \
                    split_celeba(CELEBACached.train_data, CELEBACached.train_targets,
                                 sup_frac, CELEBACached.validation_size)

            if mode == "sup":
                self.data, self.targets = CELEBACached.train_data_sup, CELEBACached.train_labels_sup
                CELEBACached.prior = torch.mean(self.targets[:, self.sub_label_inds].float(), dim=0)
            elif mode == "unsup":
                self.data = CELEBACached.train_data_unsup
                # making sure that the unsupervised labels are not available to inference
                self.targets = CELEBACached.train_labels_unsup * np.nan
            else:
                self.data, self.targets = CELEBACached.data_valid, CELEBACached.labels_valid

        else:
            self.data = self.filename
            self.targets = self.attr

        # create a batch of fixed images
        if CELEBACached.fixed_imgs is None:
            temp = []
            for i, f in enumerate(self.data[:64]):
                temp.append(self.transform(PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", f))))
            CELEBACached.fixed_imgs = torch.stack(temp, dim=0)

    def __getitem__(self, index):
        
        X = self.transform(PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.data[index])))

        target = self.targets[index].float()
        target = target[self.sub_label_inds]
        
        return X, target

    def __len__(self):
        return len(self.data)


def setup_data_loaders(use_cuda, batch_size, sup_frac=1.0, root=None, cache_data=False, **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_frac: fraction of supervised data examples
    :param cache_data: saves dataset to memory, prevents reading from file every time
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """

    if root is None:
        root = get_data_directory(__file__)
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    cached_data = {}
    loaders = {}

    #clear previous cache
    CELEBACached.clear_cache()

    if sup_frac == 0.0:
        modes = ["unsup", "test"]
    elif sup_frac == 1.0:
        modes = ["sup", "test", "valid"]
    else:
        modes = ["unsup", "test", "sup", "valid"]
        
    for mode in modes:
        cached_data[mode] = CELEBACached(root=root, mode=mode, download=True, sup_frac=sup_frac)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)
    return loaders

