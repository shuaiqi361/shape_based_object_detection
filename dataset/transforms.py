import torch
import random
import torchvision.transforms.functional as FT
from metrics import find_jaccard_overlap
import numpy as np
from PIL import Image
from PIL import ImageDraw


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 3
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims, return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param return_percent_coords: coordinates range [0, 1] or actual coordinates
    :param dims: image size after resizing
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)
    # print('In resize:', boxes)
    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def resize_keep(image, boxes, dim, return_percent_coords=True):
    """
    Resize image while keeping the orientation, the aspect ratio will change slightly. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param return_percent_coords: coordinates range [0, 1] or actual coordinates
    :param dims: image size after resizing
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    width, height = image.size
    if width > height:
        resize_factor = dim / height
        dims = (int(width * resize_factor), dim)
    else:
        resize_factor = dim / width
        dims = (dim, int(height * resize_factor))

    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, split, resize_dim, config):
    """
    Apply the transformations above.

    :param config:
    :param operation_list:
    :param resize_dim:
    :param resize: resize training images
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST', 'VAL'}
    operation_list = config.model['operation_list']
    return_percent_coords = config.model['return_percent_coords']
    # resize_dims_list = config.model['input_size']

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.3 and 'expand' in operation_list:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        if random.random() < 0.3 and 'random_crop' in operation_list:
            new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)
        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image
    new_image, new_boxes = resize(new_image, new_boxes, dims=resize_dim, return_percent_coords=return_percent_coords)
    # temp_boxes = new_boxes.clamp_(0, 1)
    # draw = ImageDraw.Draw(new_image)
    # n_boxes = temp_boxes.size(0)
    # for i in range(n_boxes):
    #     coord = ((int(temp_boxes[i][0] * resize_dim[1]), int(temp_boxes[i][1] * resize_dim[0])),
    #              (int(temp_boxes[i][2] * resize_dim[1]), int(temp_boxes[i][3] * resize_dim[0])))
    #     draw.rectangle(coord)
    # new_image.show()
    # new_image.show()
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels


def transform_richer(image, boxes, labels, split, config):
    """
    Apply the transformations above.

    :param config:
    :param operation_list:
    :param resize_dim:
    :param resize: resize training images
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST', 'VAL'}
    operation_list = config.model['operation_list']
    return_percent_coords = config.model['return_percent_coords']

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean = config.model['mean']
    std = config.model['std']

    input_sizes = config.model['input_size']  # a list of input sizes
    test_size = config.model['test_size']
    resize_dim = (test_size, test_size)

    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        if 'expand' in operation_list or 'random_crop' in operation_list:
            # Convert PIL image to Torch tensor
            new_image = FT.to_tensor(new_image)

            # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
            # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
            if random.random() < 0.25 and 'expand' in operation_list:
                new_image, new_boxes = expand(new_image, boxes, filler=mean)

            # Randomly crop image (zoom in)
            if random.random() < 0.25 and 'random_crop' in operation_list:
                new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

            # Convert Torch tensor to PIL image
            new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

        return new_image, new_boxes, new_labels  # return PIL images

    # Resize image
    new_image, new_boxes = resize(new_image, new_boxes, dims=(test_size, test_size),
                                  return_percent_coords=return_percent_coords)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels


def bof_augment(images, boxes, labels, config):
    operation_list = config.model['operation_list']
    assert len(labels) == 4  # hard code 4 images per batch, return 2 images,
    # the box is a list of 4 tensors, each tensor contains a 2-d vetor with many bboxes
    new_images = list()
    new_labels = list()
    new_boxes = list()
    resize_dims = config.model['input_size']
    if 'random_shape' in operation_list:
        resize_dim = resize_dims[np.random.randint(0, len(resize_dims))]
    else:
        if isinstance(resize_dims, list):
            resize_dim = resize_dims[0]
        else:
            resize_dim = resize_dims
    # print('boxes from loader', boxes)

    if 'mixup' in operation_list and random.random() < 0.5:
        temp_image, temp_boxes, temp_labels = mixup_image(images[:2], boxes[:2], labels[:2])
        temp_image = FT.to_pil_image(temp_image)

        temp_image, temp_boxes = resize(temp_image, temp_boxes, dims=(resize_dim, resize_dim),
                                        return_percent_coords=config.model['return_percent_coords'])
        temp_image = FT.to_tensor(temp_image)
        temp_image = torch.where(temp_image == 0,
                                 (torch.FloatTensor(config.model['mean']).unsqueeze(1).unsqueeze(1)).expand_as(temp_image),
                                 temp_image)
        temp_image = FT.normalize(temp_image, mean=config.model['mean'], std=config.model['std'])

        # fill the background with no mixup with mean pixels
        # temp_image = torch.where(temp_image == 0,
        #                          (torch.FloatTensor(config.model['mean']).unsqueeze(1).unsqueeze(1)).expand_as(temp_image),
        #                          temp_image)
        # temp_image = torch.where(temp_image[0, :, :] == 0 & temp_image[1, :, :] == 0 & temp_image[2, :, :] == 0,
        #                          (torch.FloatTensor(config.model['mean']).unsqueeze(1).unsqueeze(1)).expand_as(
        #                              temp_image),
        #                          temp_image)
        # temp_image[temp_image[:, :, :] == torch.FloatTensor([0, 0, 0])] = torch.FloatTensor(config.model['mean'])
        # temp_image = FT.to_pil_image(temp_image)

        new_images.append(temp_image)
        new_boxes.append(temp_boxes.clamp_(0, 1))
        new_labels.append(temp_labels)
    else:
        temp_image, temp_boxes = resize(images[0], boxes[0], dims=(resize_dim, resize_dim),
                                        return_percent_coords=config.model['return_percent_coords'])
        temp_image = FT.to_tensor(temp_image)
        temp_image = FT.normalize(temp_image, mean=config.model['mean'], std=config.model['std'])
        new_images.append(temp_image)
        new_labels.append(labels[0])
        new_boxes.append(temp_boxes.clamp_(0, 1))

    # # draw augmented images and bboxes
    # temp_boxes = temp_boxes.clamp_(0, 1)
    # draw = ImageDraw.Draw(temp_image)
    # n_boxes = temp_boxes.size(0)
    # rect_coord = []
    # for i in range(n_boxes):
    #     coord = ((int(temp_boxes[i][0] * resize_dim), int(temp_boxes[i][1] * resize_dim)),
    #              (int(temp_boxes[i][2] * resize_dim), int(temp_boxes[i][3] * resize_dim)))
    #     # rect_coord.append(coord)
    #     draw.rectangle(coord)
    # temp_image.show()
    #
    # exit()

    if 'mosaic' in operation_list and random.random() < 0.5:
        temp_image, temp_boxes, temp_labels = mosaic_image(images, boxes, labels)
        # temp_boxes = torch.cat(temp_boxes, dim=0)
        temp_image, temp_boxes = resize(temp_image, temp_boxes, dims=(resize_dim, resize_dim),
                                        return_percent_coords=config.model['return_percent_coords'])
        temp_image = FT.to_tensor(temp_image)
        temp_image = FT.normalize(temp_image, mean=config.model['mean'], std=config.model['std'])
        new_images.append(temp_image)
        new_boxes.append(temp_boxes.clamp_(0, 1))
        new_labels.append(temp_labels)
    else:
        temp_image, temp_boxes = resize(images[2], boxes[2], dims=(resize_dim, resize_dim),
                                        return_percent_coords=config.model['return_percent_coords'])
        temp_image = FT.to_tensor(temp_image)
        temp_image = FT.normalize(temp_image, mean=config.model['mean'], std=config.model['std'])
        new_images.append(temp_image)
        new_labels.append(labels[2])
        new_boxes.append(temp_boxes.clamp_(0, 1))

    # draw augmented images and bboxes
    # temp_boxes = temp_boxes.clamp_(0, 1)
    # draw = ImageDraw.Draw(temp_image)
    # n_boxes = temp_boxes.size(0)
    # rect_coord = []
    # for i in range(n_boxes):
    #     coord = ((int(temp_boxes[i][0] * resize_dim), int(temp_boxes[i][1] * resize_dim)),
    #              (int(temp_boxes[i][2] * resize_dim), int(temp_boxes[i][3] * resize_dim)))
    #     # rect_coord.append(coord)
    #     draw.rectangle(coord)
    # temp_image.show()
    #
    # exit()

    return new_images, new_boxes, new_labels


def mosaic_image(images, boxes, labels):
    assert len(labels) == 4  # currently only support mosaic 4 images
    new_image = Image.new('RGB', (512 * 2, 512 * 2))
    new_labels = list()
    new_boxes = list()
    # print(boxes)
    for i in range(len(labels)):
        temp_image = images[i]
        temp_boxes = boxes[i]
        temp_image, temp_boxes = resize(temp_image, temp_boxes, dims=(512, 512), return_percent_coords=False)
        new_image.paste(temp_image, box=(512 * (i % 2), 512 * (i // 2)))
        # print('Paste coords: ', 512 * (i // 2), 512 * (i % 2))
        new_labels.append(labels[i])
        temp_boxes[:, 0] += 512 * (i % 2)
        temp_boxes[:, 1] += 512 * (i // 2)
        temp_boxes[:, 2] += 512 * (i % 2)
        temp_boxes[:, 3] += 512 * (i // 2)
        new_boxes.append(temp_boxes)

    return new_image, torch.cat(new_boxes, dim=0), torch.cat(new_labels, dim=0)


def mixup_image(images, boxes, labels, beta=1.5):
    assert len(labels) == 2  # currently only support mix 2 images up, box is still a list of bboxes for 2 images
    image1 = FT.to_tensor(images[0])
    image2 = FT.to_tensor(images[1])
    _, height1, width1 = image1.size()
    _, height2, width2 = image2.size()
    # print('In mixup: ', boxes)
    new_width = max(width1, width2)
    new_height = max(height1, height2)
    new_image = torch.zeros((3, new_height, new_width))

    # lam = np.random.beta(beta, beta)
    lam = np.random.uniform(0.4, 0.6, 1)[0]

    if new_height > height1:  # sample where to put the image1
        start_idx_h = np.random.randint(0, new_height - height1)
        if new_width > width1:
            start_idx_w = np.random.randint(0, new_width - width1)
            new_image[:, start_idx_h:start_idx_h + height1, start_idx_w:start_idx_w + width1] += image1 * lam
            boxes[0][:, 0] += start_idx_w
            boxes[0][:, 1] += start_idx_h
            boxes[0][:, 2] += start_idx_w
            boxes[0][:, 3] += start_idx_h
        else:
            new_image[:, start_idx_h:start_idx_h + height1, :] += image1 * lam
            boxes[0][:, 1] += start_idx_h
            boxes[0][:, 3] += start_idx_h
    else:
        if new_width > width1:
            start_idx_w = np.random.randint(0, new_width - width1)
            new_image[:, :, start_idx_w:start_idx_w + width1] += image1 * lam
            boxes[0][:, 0] += start_idx_w
            boxes[0][:, 2] += start_idx_w
        else:
            new_image += image1 * lam

    if new_height > height2:  # sample where to put the image2
        start_idx_h = np.random.randint(0, new_height - height2)
        if new_width > width2:
            start_idx_w = np.random.randint(0, new_width - width2)
            new_image[:, start_idx_h:start_idx_h + height2, start_idx_w:start_idx_w + width2] += image2 * (1 - lam)
            boxes[1][:, 0] += start_idx_w
            boxes[1][:, 1] += start_idx_h
            boxes[1][:, 2] += start_idx_w
            boxes[1][:, 3] += start_idx_h
        else:
            new_image[:, start_idx_h:start_idx_h + height2, :] += image2 * (1 - lam)
            boxes[1][:, 1] += start_idx_h
            boxes[1][:, 3] += start_idx_h
    else:
        if new_width > width2:
            start_idx_w = np.random.randint(0, new_width - width2)
            new_image[:, :, start_idx_w:start_idx_w + width2] += image2 * (1 - lam)
            boxes[1][:, 0] += start_idx_w
            boxes[1][:, 2] += start_idx_w
        else:
            new_image += image2 * (1 - lam)

    return new_image, torch.cat([boxes[0], boxes[1]], dim=0), torch.cat([labels[0], labels[1]], dim=0)
