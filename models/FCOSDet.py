import torch.nn as nn
import torch
from math import sqrt, log

import torch.utils.model_zoo as model_zoo
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SigmoidFocalLoss
from metrics import find_jaccard_overlap
from .utils import BasicBlock, Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class PyramidFeatures(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(c5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, c3, c4, c5):
        P5_x = self.P5_1(c5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(c4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(c3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(c5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_classes + 1, kernel_size=3, padding=1)  # add centerness predictions
        # self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        # out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes x n_anchors
        out1 = out.permute(0, 2, 3, 1)

        return out1.contiguous().view(x.size(0), -1, self.num_classes)


class FCOS(nn.Module):
    """
        The Fully Convolutional One-Stage detection network - encapsulates the base ResNet network,
        Detection, and Classification Head. Without centerness branch
        """

    def __init__(self, n_classes, block, layers, prior=0.01, device='cuda:0'):
        super(FCOS, self).__init__()
        self.inplanes = 64  # dim of c3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.scale_param = nn.Parameter(torch.FloatTensor([4, 8, 16, 32, 64]))
        # self.scale_param.requires_grad = False

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.prior = prior
        self.n_classes = n_classes
        self.device = device
        self.INF = 1e6
        self.locations = self.compute_location()

        self.fmap_dims = {'c3': 64,
                          'c4': 32,
                          'c5': 16,
                          'c6': 8,
                          'c7': 4}

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError("Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-log((1.0 - self.prior) / self.prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """
        Freeze BatchNorm layers
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def compute_location(self):
        fmaps = list(self.fmap_dims.keys())
        locations = list()

        for k, fmap in enumerate(fmaps):
            locations_per_level = list()
            for i in range(self.fmap_dims[fmap]):
                for j in range(self.fmap_dims[fmap]):
                    cx = (j + 0.5) / self.fmap_dims[fmap]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / self.fmap_dims[fmap]
                    locations_per_level.append([cx, cy])

            locations.append(torch.FloatTensor(locations_per_level).to(self.device))

        assert len(locations) == 5 and locations[0].size(0) == 64 * 64 and locations[1].size(1) == 2

        return locations

    def postprocess(self, box_pred, cls_pred, center_pred):
        batch_size = cls_pred.size(0)
        # n_classes = cls_pred.size(1)
        cls_pred = cls_pred.sigmoid()  # sigmoid focal loss
        center_pred = center_pred.sigmoid()  # sigmoid binary cross entropy loss

        cls_pred = cls_pred * center_pred[:, :, None]

        locations = torch.cat(self.locations, dim=0)  # n_cells, 2
        n_cells = locations.size(0)
        predicted_locs = torch.zeros((batch_size, n_cells, 4), dtype=torch.float).to(self.config.device)
        for i in range(batch_size):
            predicted_locs[i, :, 0] = locations[:, 0] - box_pred[i, :, 0]
            predicted_locs[i, :, 1] = locations[:, 1] - box_pred[i, :, 1]
            predicted_locs[i, :, 2] = locations[:, 0] + box_pred[i, :, 2]
            predicted_locs[i, :, 3] = locations[:, 1] + box_pred[i, :, 3]

        return predicted_locs, cls_pred

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn(x2, x3, x4)

        # locs = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # class_scores = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        class_scores = list()
        locs = list()
        centerness = list()

        for i in range(len(features)):
            feat = features[i]
            cls_out = self.classificationModel(feat)
            center_out = cls_out[:, :, 0]
            cls_out = cls_out[:, :, 1:]
            bbox_out = self.regressionModel(feat)
            bbox_out = torch.sigmoid(self.scale_param[i] * bbox_out)

            class_scores.append(cls_out)
            locs.append(bbox_out)
            centerness.append(center_out)

        locs = torch.cat(locs, dim=1)
        class_scores = torch.cat(class_scores, dim=1)
        centerness = torch.cat(centerness, dim=1)

        return locs, class_scores, centerness


class FCOSLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, locations, config, threshold=0.5, center_sample=True):
        super(FCOSLoss, self).__init__()
        self.threshold = threshold
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.config = config
        self.locations = locations
        self.center_sample = center_sample
        self.INF = 1e6

        self.Diou_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        self.Focal_loss = SigmoidFocalLoss(gamma=2.0, alpha=0.25, config=config)
        self.center_loss = nn.BCEWithLogitsLoss()
        self.fpn_strides = [8 / 512., 16 / 512., 32 / 512., 64 / 512., 128 / 512.]
        self.sizes = [[0., 0.08], [0.08, 0.16], [0.16, 0.32], [0.32, 0.64], [0.64, 1.]]
        self.radius = 1.5

    def increase_threshold(self, increment=0.1):
        if self.threshold >= 0.7:
            return

        self.threshold += increment

    def prepare_targets(self, locations, boxes, labels):
        valid_sizes = []
        n_locations_per_level = []
        batch_size = len(labels)

        for i, locations_per_level in enumerate(locations):  # list of length 5, each element is a tensor of n, 2
            size_per_level = locations_per_level.new_tensor(self.sizes[i])
            valid_sizes.append(size_per_level[None].expand(len(locations_per_level), -1))
            n_locations_per_level.append(len(locations_per_level))

        valid_sizes = torch.cat(valid_sizes, dim=0)  # cat valid sizes for all levels of features
        all_locations = torch.cat(locations, dim=0)  # n_cells, 2

        label_targets_all = list()
        bbox_targets_all = list()
        for i in range(batch_size):
            label_target, bbox_target = self.assign_targets(all_locations, boxes[i],
                                                            labels[i], valid_sizes, n_locations_per_level)
            # split the labels for each image according to the level of feature pyramid
            label_target = torch.split(label_target, n_locations_per_level, dim=0)
            bbox_target = torch.split(bbox_target, n_locations_per_level, dim=0)
            label_targets_all.append(label_target)
            bbox_targets_all.append(bbox_target)

        labels_batch = list()
        bboxes_batch = list()

        for level in range(len(locations)):
            # cat all labels and bboxes, list of five (batch_size x n_cells/level, )
            labels_batch.append(torch.cat([label_per_image[level] for label_per_image in label_targets_all], dim=0))
            bboxes_batch.append(torch.cat([bbox_per_image[level] for bbox_per_image in bbox_targets_all], dim=0))

        return labels_batch, bboxes_batch

    def assign_targets(self, locations, boxes, labels, valid_sizes, n_locations_per_level):
        """
        :param locations: (n_cells, 2) all possible locations on different levels of feature maps
        :param boxes: n_boxes, 4, x1y2, x2y2
        :param labels: n_boxes,
        :param valid_sizes: list of 5 lists(range)
        :param n_locations_per_level: list of integers, indicating number of cells on each pyramidal level
        :return:
        """
        assert locations.size(1) == 2

        locs_x, locs_y = locations[:, 0], locations[:, 1]  # (n_cells,)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (n_cells,)
        if areas.size(0) > 1:
            areas, area_sort_id = areas.sort(dim=0, descending=True)
            labels = labels[area_sort_id]

        # init targets
        l = locs_x[:, None] - boxes[:, 0][None]  # (n_cells, 1) - (1, n_boxes) = (n_cells, n_boxes)
        t = locs_y[:, None] - boxes[:, 1][None]
        r = boxes[:, 2][None] - locs_x[:, None]
        b = boxes[:, 3][None] - locs_y[:, None]  # (n_cells, n_boxes)

        bbox_targets = torch.stack([l, t, r, b], dim=2)  # (n_cells, n_boxes, 4)

        if self.center_sample:
            within_bbox_range = self.get_sample_region(boxes, n_locations_per_level, locations)
        else:
            within_bbox_range = bbox_targets.min(dim=2)[0] > 0  # min of targets should be positive, (n_cells, n_boxes)

        max_bbox_target = bbox_targets.max(dim=2)[0]  # find the max of the 4 targets to choose level of features

        valid_in_level = (max_bbox_target >= valid_sizes[:, 0]) \
                         and (max_bbox_target <= valid_sizes[:, 1])  # (n_cells,)

        locs_gt_area = areas[None].repeat(locations.size(0), 1)  # (n_cells, n_boxes)
        locs_gt_area[within_bbox_range == 0] = self.INF  # assigned to background
        locs_gt_area[valid_in_level == 0] = self.INF  # assigned to other levels of feature

        locs_min_area, locs_gt_idx = locs_gt_area.min(dim=1)  # find the target from the minimum enclosing bbox
        bbox_targets = bbox_targets[:, locs_gt_idx]  # (n_cells, 4)
        label_targets = labels[locs_gt_idx]
        label_targets[locs_min_area == self.INF] = 0

        return label_targets, bbox_targets

    def get_sample_region(self, bboxes, n_locations_per_level, locations):
        """
        For each image in a batch
        :param bboxes: (n_bboxes, 4), #objects in one image
        :param strides:
        :param n_locations_per_level:
        :param locations:
        :param radius:
        :return:
        """
        locs_x, locs_y = locations[:, 0], locations[:, 1]
        n_objects = bboxes.size(0)
        n_locs = len(locs_x)
        bboxes = bboxes[None].expand(n_locs, n_objects, 4)
        cx = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2.
        cy = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2.

        if cx[:, :, 0].sum() == 0:
            return locs_x.new_zeros(locs_x.shape, dtype=torch.uint8)

        center_bbox = bboxes.new_zeros(bboxes.shape)

        # identify the radius for assigning ground truth boxes
        begin_idx = 0
        for level, n_cells in enumerate(n_locations_per_level):
            end_idx = begin_idx + n_cells
            stride = self.fpn_strides * self.radius

            x_min = cx[begin_idx:end_idx] - stride  # bboxes defined by the strides
            y_min = cy[begin_idx:end_idx] - stride
            x_max = cx[begin_idx:end_idx] + stride
            y_max = cy[begin_idx:end_idx] + stride
            # compare radius with bboxes and adjust the new ground truth
            center_bbox[begin_idx:end_idx, :, 0] = torch.where(x_min > bboxes[begin_idx:end_idx, :, 0],
                                                               x_min, bboxes[begin_idx:end_idx, :, 0])
            center_bbox[begin_idx:end_idx, :, 1] = torch.where(y_min > bboxes[begin_idx:end_idx, :, 1],
                                                               y_min, bboxes[begin_idx:end_idx, :, 1])
            center_bbox[begin_idx:end_idx, :, 2] = torch.where(x_max < bboxes[begin_idx:end_idx, :, 2],
                                                               x_max, bboxes[begin_idx:end_idx, :, 2])
            center_bbox[begin_idx:end_idx, :, 3] = torch.where(y_max < bboxes[begin_idx:end_idx, :, 3],
                                                               y_max, bboxes[begin_idx:end_idx, :, 3])

            begin_idx = end_idx

        l = locs_x[:, None] - center_bbox[:, :, 0]
        t = locs_y[:, None] - center_bbox[:, :, 1]
        r = center_bbox[:, :, 2] - locs_x[:, None]
        b = center_bbox[:, :, 3] - locs_y[:, None]

        bbox_targets = torch.stack([l, t, r, b], dim=2)

        within_bbox_range = bbox_targets.min(dim=2)[0] > 0

        return within_bbox_range

    def compute_centerness_targets(self, bboxes_targets_flat):
        left_right = bboxes_targets_flat[:, [0, 2]]
        top_bottom = bboxes_targets_flat[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
                top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

    def forward(self, predicted_locs, predicted_scores, predicted_centerness, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 22536 prior boxes, a tensor of dimensions (N, 22536, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 22536, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_classes = predicted_scores.size(2)

        assert predicted_locs.size(1) == predicted_scores.size(1)

        labels_batch, bboxes_batch = self.prepare_targets(self.locations, boxes, labels)

        pred_labels_flat = list()
        pred_bboxes_flat = list()
        pred_center_flat = list()
        labels_batch_flat = list()
        bboxes_batch_flat = list()

        for i in range(batch_size):
            pred_labels_flat.append(predicted_scores[i].view(-1, n_classes))
            pred_bboxes_flat.append(predicted_locs[i].view(-1, 4))
            pred_center_flat.append(predicted_centerness[i].view(-1))

            labels_batch_flat.append(labels_batch[i].view(-1))
            bboxes_batch_flat.append(bboxes_batch[i].view(-1, 4))

        pred_labels_flat = torch.cat(pred_labels_flat, dim=0)
        pred_bboxes_flat = torch.cat(pred_bboxes_flat, dim=0)
        pred_center_flat = torch.cat(pred_center_flat, dim=0)
        labels_batch_flat = torch.cat(labels_batch_flat, dim=0)
        bboxes_batch_flat = torch.cat(bboxes_batch_flat, dim=0)

        positives_idx = torch.nonzero(labels_batch_flat > 0).squeeze(1)

        conf_loss = self.Focal_loss(pred_labels_flat,
                                    labels_batch_flat.int()) / (
                            positives_idx.numel() + batch_size)  # in case no positives

        positives_pred_bboxes = pred_bboxes_flat[positives_idx]
        positives_pred_center = pred_center_flat[positives_idx]
        positivs_target_bboxes = bboxes_batch_flat[positives_idx]

        if positives_idx.numel() > 0:
            center_targets = self.compute_centerness_targets(positivs_target_bboxes)
            loc_loss = self.Diou_loss(positives_pred_bboxes, positivs_target_bboxes, weights=center_targets)
            center_loss = self.center_loss(positives_pred_center, center_targets)
        else:
            loc_loss = positives_pred_bboxes.sum()
            center_loss = positives_pred_center.sum()

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss + center_loss


def resnet50_fcos(num_classes, config, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param config:
        :param pretrained:
        :param num_classes:
    """
    model = FCOS(num_classes, Bottleneck, [3, 4, 6, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101_fcos(num_classes, config, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param config:
        :param pretrained:
        :param num_classes:
    """
    model = FCOS(num_classes, Bottleneck, [3, 4, 23, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model
