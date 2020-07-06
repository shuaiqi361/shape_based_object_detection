import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from .transforms import transform, transform_richer, transform_traffic
import torchvision.transforms.functional as FT


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        # self.input_size = input_size
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
        # image, boxes, labels, = transform_richer(image, boxes, labels,
        #                                          split=self.split,
        #                                          config=self.config)
        image, boxes, labels, = transform(image, boxes, labels,
                                          split=self.split,
                                          config=self.config)

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

    def __init__(self, data_folder, split, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        # self.input_size = input_size
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
        # image, boxes, labels = transform_richer(image, boxes, labels,
        #                                         split=self.split,
        #                                         config=self.config)
        image, boxes, labels = transform(image, boxes, labels,
                                         split=self.split,
                                         config=self.config)
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

    def __init__(self, data_folder_list, split, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        # self.input_size = input_size
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        self.config = config
        self.data_folder_list = data_folder_list.split(' ')
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
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        ids = objects['image_id']
        difficulties = torch.LongTensor(objects['difficulties'])

        # Apply transformations
        # image, boxes, labels = transform(image, boxes, labels,
        #                                  split=self.split,
        #                                  resize_dim=self.input_size, config=self.config)
        image, boxes, labels = transform_richer(image, boxes, labels,
                                                split=self.split,
                                                config=self.config)

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

        # images = torch.stack(images, dim=0)

        return images, boxes, labels, ids, difficulties


class DetracDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Input: list of traffic dataset json files, UA-DETRAC dataset
    """

    def __init__(self, data_folder_list, split, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        self.config = config
        self.data_folder_list = data_folder_list
        self.images = list()
        self.objects = list()

        # Read data files
        with open(os.path.join(self.data_folder_list, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(self.data_folder_list, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # image = Image.open(self.images[i], mode='r')
        # image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        ids = objects['image_id']
        difficulties = torch.LongTensor(objects['difficulties'])
        if len(objects['ignore_regions']) > 0:
            ignored_regions = torch.FloatTensor(objects['ignore_regions'])
        else:
            ignored_regions = torch.FloatTensor([[0.5, 0.5, 0.5, 0.5]])
        occlusions = objects['occlusions']

        # Read image, and remove ignored regions
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # img = cv2.imread(self.images[i])

        # temporarily using ignored regions for training
        # for region in ignore_regions:
        #     cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (127, 127, 127), -1)

        # cv2.imshow('inputs', img)
        # cv2.waitKey()
        # exit()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(img)

        # Apply transformations
        # image, boxes, labels = transform(image, boxes, labels,
        #                                  split=self.split, resize_dim=(540, 960),
        #                                  config=self.config)
        image, boxes, labels, ignored_regions = transform_traffic(image, boxes, labels, ignored_regions,
                                                                  split=self.split,
                                                                  config=self.config)

        return image, boxes, labels, ignored_regions, ids, difficulties

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
        ignored_regions = list()
        ids = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            ignored_regions.append(b[3])
            ids.append(b[4])
            difficulties.append(b[5])

        # images = torch.stack(images, dim=0)

        return images, boxes, labels, ignored_regions, ids, difficulties


class BaseModelVOCOCODataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Input: list of json files: coco and voc datasets
    """

    def __init__(self, data_folder_list, split, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        self.config = config
        self.data_folder_list = data_folder_list.split(' ')
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
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        if 'image_id' in objects.keys():
            ids = objects['image_id']
        else:
            ids = -1
        difficulties = torch.LongTensor(objects['difficulties'])

        # Apply transformations
        image, boxes, labels = transform_richer(image, boxes, labels,
                                                split=self.split,
                                                config=self.config)

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

        # images = torch.stack(images, dim=0)

        return images, boxes, labels, ids, difficulties


class COCOMultiScaleDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, config):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        # self.input_size = input_size
        assert config is not None
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        self.config = config
        self.data_folder = data_folder
        self.min_side = config.model['shorter_side']
        self.max_side = config.model['longer_side']
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

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
                                         split=self.split,
                                         config=self.config)
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

        temp_imgs = [b[0] for b in batch]

        widths = [int(img.size(1)) for img in temp_imgs]
        heights = [int(img.size(2)) for img in temp_imgs]

        max_width = max(widths)
        max_height = max(heights)
        batch_size = len(batch)

        padded_images = torch.zeros(batch_size, 3, max_width, max_height)

        for i in range(batch_size):
            img = temp_imgs[i]
            padded_images[i, :, :img.size(1), :img.size(2)] = img  # no need to adjust the bbox coordinates

        if max_height < max_width:
            scale = self.min_side / max_height
        else:
            scale = self.min_side / max_width
        for i in range(batch_size):
            # adjust the size of the image and gt bboxes
            img = padded_images[i]
            bbox = [[bb[0] / max_width, bb[1] / max_height, bb[2] / max_width,
                     bb[3] / max_height] for bb in batch[i][1]]  # change coordinates to percentages

            new_image = FT.resize(img, (int(round(max_width * scale)), int(round(max_height * scale))))
            # new_image = FT.to_tensor(new_image)
            # new_image = FT.normalize(new_image, mean=self.mean, std=self.std)

            images.append(new_image)
            boxes.append(bbox)
            labels.append(batch[i][2])
            ids.append(batch[i][3])
            difficulties.append(batch[i][4])

            # draw augmented images and bboxes
            temp_boxes = [[bb[0] * int(round(max_width * scale)), bb[1] * int(round(max_height * scale)),
                           bb[2] * int(round(max_width * scale)),
                           bb[3] * int(round(max_height * scale))] for bb in bbox]
            draw = Image.ImageDraw.Draw(new_image)
            n_boxes = temp_boxes.size(0)
            for j in range(n_boxes):
                coord = ((temp_boxes[j][0], temp_boxes[j][1]),
                         (temp_boxes[j][2], temp_boxes[j][3]))
                draw.rectangle(coord)
            new_image.show()

            exit()

        images = torch.stack(images, dim=0)

        return images, boxes, labels, ids, difficulties
