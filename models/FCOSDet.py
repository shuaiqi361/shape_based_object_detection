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

        for i, locations_per_level in enumerate(locations):
            size_per_level = locations_per_level.new_tensor(self.sizes[i])
            valid_sizes.append(size_per_level[None].expand(len(locations_per_level), -1))
            n_locations_per_level.append(len(locations_per_level))

        valid_sizes = torch.cat(valid_sizes, dim=0)  # cat valid sizes for all levels of features
        all_locations = torch.cat(locations, dim=0)

        labels, bbox_targets = self.assign_targets(all_locations, boxes, labels, valid_sizes, n_locations_per_level)


    def assign_targets(self, locations, boxes, labels, valid_sizes, n_locations_per_level):
        labels = list()
        bbox_targets = list()
        locs_x, locs_y = locations[:, 0], locations[:, 1]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        l = locs_x - boxes[]



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
        n_classes = predicted_scores.size(2)  # not including background

        # print(n_priors, predicted_locs.size(), predicted_scores.size())
        assert predicted_locs.size(1) == predicted_scores.size(1)

        decoded_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)
        true_neg_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)


        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss


def resnet50(num_classes, config, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param config:
        :param pretrained:
        :param num_classes:
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 6, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, config, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param config:
        :param pretrained:
        :param num_classes:
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 23, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model
