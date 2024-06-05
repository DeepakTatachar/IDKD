import torch
import numpy as np

class ProxySet(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        labels,
        idxs,
        transform,
        logger):
        self.transform = transform
        self.data = data
        self.labels = labels
        self.idxs = idxs
        self.logger = logger

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, _ =  self.data.__getitem__(self.idxs[index])
        target = self.labels[self.idxs[index]]
        target = torch.Tensor(target)
        # self.logger.info(sample)
        return sample, target


class OneHotLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes, logger) -> None:
        self.dataset = dataset
        self.num_classes = num_classes
        self.logger = logger

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        sample, target = self.dataset.__getitem__(index)
        label = torch.zeros(self.num_classes)
        label[target] = 1
        return sample, label
    
class ProxySet2(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        labels,
        idxs,
        transform,
        logger):
        self.transform = transform
        self.data = data
        self.labels = labels
        self.idxs = idxs
        self.logger = logger

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, _ =  self.data.__getitem__(self.idxs[index])
        target = self.labels[index]
        target = torch.Tensor(target)
        # self.logger.info(sample)
        return sample, target