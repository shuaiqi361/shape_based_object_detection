import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from .transforms import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, input_size, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.input_size = input_size
        assert config is not None
        self.config = config
        assert self.split in {'TRAIN', 'TEST', 'VAL'}

        self.data_folder = data_folder

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.LongTensor(objects['difficulties'])  # (n_objects)
        ids = self.images[i]

        # Apply transformations
        image, boxes, labels, = transform(image, boxes, labels,
                                          split=self.split, resize_dim=self.input_size,
                                          operation_list=self.config.model['operation_list'])

        return image, boxes, labels, ids, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function
        (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        ids = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            ids.append(b[3])
            difficulties.append(b[4])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, ids, difficulties


class COCO17Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, input_size, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.input_size = input_size
        assert config is not None
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        self.config = config
        self.data_folder = data_folder

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        ids = objects['image_id']
        difficulties = torch.LongTensor(objects['difficulties'])  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels,
                                         split=self.split, resize_dim=self.input_size,
                                         operation_list=self.config.model['operation_list'])
        return image, boxes, labels, ids, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function
        (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        ids = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            ids.append(b[3])
            difficulties.append(b[4])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, ids, difficulties


class TrafficDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Input: list of traffic dataset json files
    """

    def __init__(self, data_folder_list, split, input_size, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.input_size = input_size
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        self.config = config
        self.data_folder_list = data_folder_list.split(' ')
        self.input_size = input_size
        self.images = list()
        self.objects = list()

        if isinstance(self.data_folder_list, list):
            for data_folder in self.data_folder_list:
                # Read data files, skip data with no validation split set
                if not os.path.exists(os.path.join(data_folder, self.split + '_images.json')):
                    continue
                with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
                    self.images += json.load(j)
                with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
                    self.objects += json.load(j)
        else:
            # Read data files
            with open(os.path.join(self.data_folder_list, self.split + '_images.json'), 'r') as j:
                self.images = json.load(j)
            with open(os.path.join(self.data_folder_list, self.split + '_objects.json'), 'r') as j:
                self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        ids = self.images[i]

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels,
                                         split=self.split,
                                         resize_dim=self.input_size, operation_list=self.config.operation_list)

        return image, boxes, labels, ids

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function
        (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        ids = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            ids.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, ids
