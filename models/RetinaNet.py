import torch.nn as nn
import torch
from math import sqrt, log

import torch.utils.model_zoo as model_zoo
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, focal_loss
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

        return P3_x, P4_x, P5_x, P6_x, P7_x


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

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
    def __init__(self, num_features_in, num_anchors, num_classes, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

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
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes x n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RetinaNet(nn.Module):
    """
        The RetinaNet network - encapsulates the base ResNet network, Detection, and Classification Head.
        """

    def __init__(self, n_classes, block, layers, prior=0.01, device='cuda:0'):
        super(RetinaNet, self).__init__()
        self.inplanes = 64  # dim of c3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.prior = prior
        self.n_classes = n_classes
        self.device = device

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.anchors_cxcy = self.create_anchors()
        self.priors_cxcy = self.anchors_cxcy

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256, num_anchors=9)
        self.classificationModel = ClassificationModel(256, num_anchors=9, num_classes=n_classes)

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

        # self.freeze_bn()

    def create_anchors(self):
        """
        Create the anchor boxes as in RetinaNet
        :return: prior boxes in center-size coordinates
        """
        fmap_dims = {'c3': 64,
                     'c4': 32,
                     'c5': 16,
                     'c6': 8,
                     'c7': 4}

        obj_scales = {'c3': 0.0625,
                      'c4': 0.125,
                      'c5': 0.25,
                      'c6': 0.5,
                      'c7': 1.}
        scale_factor = [2. ** 0, 2. ** (1 / 3.), 2. ** (2 / 3.)]
        aspect_ratios = {'c3': [1., 2., 0.5],
                         'c4': [1., 2., 0.5],
                         'c5': [1., 2., 0.5],
                         'c6': [1., 2., 0.5],
                         'c7': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        anchors = list()

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        for fac in scale_factor:
                            anchors.append([cx, cy, obj_scales[fmap] * fac * sqrt(ratio),
                                            obj_scales[fmap] * fac / sqrt(ratio)])

        anchors = torch.FloatTensor(anchors).to(self.device)  # (49104, 4)

        return anchors.clamp_(0, 1)

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

        # print(x2.size(), x3.size(), x4.size())
        # print('Shapes for FPN:')
        # for feat in features:
        #     print(feat.size())
        #
        # exit()

        locs = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        class_scores = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        return locs, class_scores


class RetinaFocalLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, threshold=0.5, neg_pos_ratio=3):
        super(RetinaFocalLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.config = config

        self.smooth_l1 = nn.L1Loss()
        self.Diou_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        # self.Focal_loss = FocalLoss(class_num=self.n_classes, size_average=True)
        self.Focal_loss = focal_loss

    def increase_threshold(self, increment=0.1):
        if self.threshold >= 0.7:
            return

        self.threshold += increment

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 22536 prior boxes, a tensor of dimensions (N, 22536, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 22536, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        # print(n_priors, predicted_locs.size(), predicted_scores.size())
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        decoded_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)
        true_neg_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 22536)

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (22536)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior_neg_used = labels[i][object_for_each_prior]

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            # label_for_each_prior[overlap_for_each_prior < self.threshold] = -1  # label in 0.4-0.5 is not used
            # label_for_each_prior[overlap_for_each_prior < self.threshold - 0.1] = 0
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            label_for_each_prior_neg_used[overlap_for_each_prior < self.threshold - 0.1] = -1

            # Store
            true_classes[i] = label_for_each_prior
            true_neg_classes[i] = label_for_each_prior_neg_used

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
            true_locs[i] = boxes[i][object_for_each_prior]
            decoded_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0
        negative_priors = true_neg_classes == -1

        # LOCALIZATION LOSS
        if self.config.reg_loss.upper() == 'DIOU':
            loc_loss = self.Diou_loss(decoded_locs[positive_priors].view(-1, 4),
                                      true_locs[positive_priors].view(-1, 4))
        else:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors].view(-1, 4),
                                      true_locs_encoded[positive_priors].view(-1, 4))

        # CONFIDENCE LOSS
        if self.config.cls_loss.upper() == 'FOCAL':
            predicted_objects = torch.cat([predicted_scores[positive_priors],
                                           predicted_scores[negative_priors]], dim=0)
            target_class = torch.cat([true_classes[positive_priors],
                                      true_classes[negative_priors]], dim=0)
            conf_loss = self.Focal_loss(predicted_objects.view(-1, n_classes),
                                        target_class.view(-1), device=self.device)
        else:
            # Number of positive and hard-negative priors per image
            # print('Classes:', self.n_classes, predicted_scores.size(), true_classes.size())
            n_positives = positive_priors.sum(dim=1)  # (N)
            n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

            # First, find the loss for all priors
            conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes),
                                               true_classes.view(-1))  # (N * 8732)
            conf_loss_all = conf_loss_all.view(batch_size, -1)  # (N, 8732)

            # We already know which priors are positive
            conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

            # Next, find which priors are hard-negative
            # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
            # conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            conf_loss_neg = conf_loss_all[negative_priors]
            # print(positive_priors.size(), negative_priors.size(), conf_loss_pos.size(), conf_loss_neg.size())
            # conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            conf_loss_neg, _ = conf_loss_neg.sort(dim=0, descending=True)  # (N, 8732), sorted by decreasing hardness
            # hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
            #     self.device)  # (N, 8732)
            # hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            # conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))
            conf_loss_hard_neg = conf_loss_neg[:n_hard_negatives.sum().long()]

            # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
            conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

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
