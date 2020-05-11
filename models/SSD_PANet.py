from torch import nn
from operators.Deformable_convolution import DeformConv2d
import torch.nn.functional as F
from math import sqrt
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SmoothL1Loss, LabelSmoothingLoss, SigmoidFocalLoss
from metrics import find_jaccard_overlap
from .modules import mish, AdaptivePooling, AttentionHead, AttentionHeadSplit


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps
    Feel free to substitute with other pre-trained backbones
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = AdaptivePooling(256, 256, adaptive_size=64)  # for multi-scale training,
        # pool feature map of arbitrary size to 64 x 64

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = mish(self.conv1_1(image))  # (N, 64, 512, 512)
        out = mish(self.conv1_2(out))  # (N, 64, 512, 512)
        out = self.pool1(out)  # (N, 64, 256, 256)

        out = mish(self.conv2_1(out))  # (N, 128, 256, 256)
        out = mish(self.conv2_2(out))  # (N, 128, 256, 256)
        out = self.pool2(out)  # (N, 128, 128, 128)

        out = mish(self.conv3_1(out))  # (N, 256, 128, 128)
        out = mish(self.conv3_2(out))  # (N, 256, 128, 128)
        out = mish(self.conv3_3(out))  # (N, 256, 128, 128)
        out = self.pool3(out)  # (N, 256, 64, 64), it would have been 37 if not for ceil_mode = True

        out = mish(self.conv4_1(out))  # (N, 512, 64, 64)
        out = mish(self.conv4_2(out))  # (N, 512, 64, 64)
        out = mish(self.conv4_3(out))  # (N, 512, 64, 64)
        conv4_3_feats = out  # (N, 512, 64, 64)
        out = self.pool4(out)  # (N, 512, 32, 32)

        out = mish(self.conv5_1(out))  # (N, 512, 32, 32)
        out = mish(self.conv5_2(out))  # (N, 512, 32, 32)
        out = mish(self.conv5_3(out))  # (N, 512, 32, 32)
        conv5_3_feats = out  # (N, 512, 32, 32)
        out = self.pool5(out)  # (N, 512, 16, 16)

        out = mish(self.conv6(out))  # (N, 1024, 16, 16)

        conv7_feats = mish(self.conv7(out))  # (N, 1024, 16, 16)

        return conv4_3_feats, conv5_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("Loading pre-trained VGG16 base model and PANet Neck.")


class PathAggregateFeatures(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super(PathAggregateFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.N5_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.N4_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.N3_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(c5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N6_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N6_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = mish
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N7_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.N7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.mish = mish

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, c3, c4, c5):
        P5_x = self.P5_1(c5)  # [N, 1024, 16, 16]
        P5_upsampled_x = self.P5_upsampled(P5_x)  # [N, 1024, 32, 32]
        P5_x = self.P5_2(P5_x)  # [N, 256, 16, 16]

        P4_x = self.P4_1(c4)  # [N, 512, 32, 32]
        P4_x = P5_upsampled_x + P4_x  # [N, 512, 32, 32]
        P4_upsampled_x = self.P4_upsampled(P4_x)  # [N, 512, 64, 64]
        P4_x = self.P4_2(P4_x)  # [N, 256, 32, 32]

        P3_x = self.P3_1(c3)  # [N, 256, 64, 64]
        P3_x = P3_x + P4_upsampled_x  # [N, 256, 64, 64]
        P3_x = self.P3_2(P3_x)  # [N, 256, 64, 64]

        P6_x = self.P6(c5)  # [N, 256, 8, 8]

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)  # [N, 256, 4, 4]

        N3_x = self.N3_2(self.mish(P3_x))
        N4_x = self.N3_1(N3_x)
        N4_x = self.N4_2(N4_x + P4_x)

        N5_x = self.N5_1(N4_x)
        N5_x = self.N5_2(N5_x + P5_x)

        N6_x = self.N6_1(N5_x)
        N6_x = self.N6_2(N6_x + P6_x)

        N7_x = self.P7_1(N6_x)
        N7_x = self.N7_2(N7_x + P7_x)

        return N3_x, N4_x, N5_x, N6_x, N7_x


class PredictionHead(nn.Module):
    def __init__(self, n_classes, mode='full'):
        super(PredictionHead, self).__init__()

        self.n_classes = n_classes
        self.n_boxes = {
            'N3': 9,
            'N4': 9,
            'N5': 9,
            'N6': 9,
            'N7': 9
        }

        if mode.upper() == 'FULL':
            self.det_N3 = AttentionHead(inplanes=256, reg_out=self.n_boxes['N3'] * 4,
                                        cls_out=self.n_boxes['N3'] * self.n_classes)
            self.det_N4 = AttentionHead(inplanes=256, reg_out=self.n_boxes['N4'] * 4,
                                        cls_out=self.n_boxes['N4'] * self.n_classes)
            self.det_N5 = AttentionHead(inplanes=256, reg_out=self.n_boxes['N5'] * 4,
                                        cls_out=self.n_boxes['N5'] * self.n_classes)
            self.det_N6 = AttentionHead(inplanes=256, reg_out=self.n_boxes['N6'] * 4,
                                        cls_out=self.n_boxes['N6'] * self.n_classes)
            self.det_N7 = AttentionHead(inplanes=256, reg_out=self.n_boxes['N7'] * 4,
                                        cls_out=self.n_boxes['N7'] * self.n_classes)
        else:
            self.det_N3 = AttentionHeadSplit(inplanes=256, reg_out=self.n_boxes['N3'] * 4,
                                             cls_out=self.n_boxes['N3'] * self.n_classes)
            self.det_N4 = AttentionHeadSplit(inplanes=256, reg_out=self.n_boxes['N4'] * 4,
                                             cls_out=self.n_boxes['N4'] * self.n_classes)
            self.det_N5 = AttentionHeadSplit(inplanes=256, reg_out=self.n_boxes['N5'] * 4,
                                             cls_out=self.n_boxes['N5'] * self.n_classes)
            self.det_N6 = AttentionHeadSplit(inplanes=256, reg_out=self.n_boxes['N6'] * 4,
                                             cls_out=self.n_boxes['N6'] * self.n_classes)
            self.det_N7 = AttentionHeadSplit(inplanes=256, reg_out=self.n_boxes['N7'] * 4,
                                             cls_out=self.n_boxes['N7'] * self.n_classes)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                if c.bias is not None:
                    nn.init.constant_(c.bias, val=0)

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

        obj_scales = {'c3': 0.04,
                      'c4': 0.08,
                      'c5': 0.16,
                      'c6': 0.32,
                      'c7': 0.64}
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

    def forward(self, n3, n4, n5, n6, n7):
        loc_n3, cls_n3 = self.det_N3(n3)
        loc_n4, cls_n4 = self.det_N4(n4)
        loc_n5, cls_n5 = self.det_N5(n5)
        loc_n6, cls_n6 = self.det_N6(n6)
        loc_n7, cls_n7 = self.det_N7(n7)

        locs = torch.cat([loc_n3, loc_n4, loc_n5, loc_n6, loc_n7], dim=1).contiguous()
        classes_scores = torch.cat([cls_n3, cls_n4, cls_n5, cls_n6, cls_n7], dim=1).contiguous()

        return locs, classes_scores


class MultiBoxPANetLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, threshold=0.4, neg_pos_ratio=3):
        super(MultiBoxPANetLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.config = config

        self.smooth_l1 = SmoothL1Loss()
        self.Iou_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Ciou')
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.SoftCE = LabelSmoothingLoss(classes=self.n_classes, smoothing=0.05)
        self.Focal_loss = SigmoidFocalLoss(gamma=2., alpha=0.25, config=self.config)

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

        decoded_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 22536, 4)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 22536, 4)
        true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 22536, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 22536)
        true_neg_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 22536)

        # For each image
        for i in range(batch_size):
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 22536)

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (22536)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            overlap_for_each_object, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            prior_for_each_object = prior_for_each_object[overlap_for_each_object > 0]

            if len(prior_for_each_object) > 0:
                overlap_for_each_prior.index_fill_(0, prior_for_each_object, 1.0)

            for j in range(prior_for_each_object.size(0)):
                object_for_each_prior[prior_for_each_object[j]] = j

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior_neg_used = labels[i][object_for_each_prior]

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            label_for_each_prior_neg_used[overlap_for_each_prior < self.threshold - 0.1] = -1

            # Store
            true_classes[i] = label_for_each_prior
            true_neg_classes[i] = label_for_each_prior_neg_used

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = boxes[i][object_for_each_prior]
            decoded_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0
        negative_priors = true_neg_classes == -1

        # LOCALIZATION LOSS
        if self.config.reg_loss.upper() == 'IOU':
            loc_loss = self.Iou_loss(decoded_locs[positive_priors].view(-1, 4),
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
            if self.config.cls_loss.upper() == 'SOFTLABEL':
                conf_loss_all = self.SoftCE(predicted_scores.view(-1, n_classes), true_classes.view(-1))
            else:
                conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes),
                                                   true_classes.view(-1))
            conf_loss_all = conf_loss_all.view(batch_size, -1)

            # We already know which priors are positive
            conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

            # Next, find which priors are hard-negative
            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            conf_loss_neg[
                ~negative_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            # conf_loss_neg[positive_priors] = 0.
            conf_loss_neg, _ = conf_loss_neg.sort(dim=-1, descending=True)  # (N, 8732), sorted by decreasing hardness
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
                self.device)  # (N, 8732)
            hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

            conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
