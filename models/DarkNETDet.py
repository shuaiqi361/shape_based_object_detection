from torch import nn
import torch.nn.functional as F
from math import sqrt
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SmoothL1Loss, LabelSmoothingLoss, SigmoidFocalLoss
from metrics import find_jaccard_overlap
from .modules import Mish, ConvGNAct
import math


class Residual(nn.Module):
    def __init__(self, in_planes):
        super(Residual, self).__init__()
        self.in_planes = in_planes
        self.Conv1 = ConvGNAct(self.in_planes, self.in_planes // 2, kernel_size=1, padding=0)
        self.Conv2 = ConvGNAct(self.in_planes // 2, self.in_planes, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.Conv1(x)
        x = self.Conv2(x)

        return x + res


class DarkBlock(nn.Module):
    def __init__(self, in_planes, n_blocks):
        super(DarkBlock, self).__init__()
        self.in_planes = in_planes
        self.n_blocks = n_blocks
        self.Block = self.__make_block__(self.n_blocks)

    def __make_block__(self, n_blocks):
        block = []
        for i in range(n_blocks):
            layer = Residual(self.in_planes)
            block.append(layer)

        return nn.Sequential(*block)

    def forward(self, x):
        feats = self.Block(x)
        return feats


class DarknetBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps
    Feel free to substitute with other pre-trained backbones
    """

    def __init__(self):
        super(DarknetBase, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.GN1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)  # (512, 512)
        self.GN2 = nn.GroupNorm(16, 64)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # (256, 256)
        self.GN3 = nn.GroupNorm(16, 128)

        self.DarkB1 = DarkBlock(128, 3)
        self.transit_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # (128, 128)
        self.GN_t1 = nn.GroupNorm(32, 256)

        self.DarkB2 = DarkBlock(256, 4)
        self.transit_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)  # (64, 64)
        self.GN_t2 = nn.GroupNorm(32, 512)

        self.DarkB3 = DarkBlock(512, 4)
        self.transit_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)  # (32, 32)
        self.GN_t3 = nn.GroupNorm(32, 512)

        self.DarkB4 = DarkBlock(512, 3)
        self.transit_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=2)  # (16, 16)
        self.GN_t4 = nn.GroupNorm(32, 256)

        self.DarkB5 = DarkBlock(256, 3)
        self.transit_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # (8, 8)
        self.GN_t5 = nn.GroupNorm(32, 256)

        self.DarkB6 = DarkBlock(256, 2)
        self.transit_6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # (4, 4)
        self.GN_t6 = nn.GroupNorm(32, 256)

        self.DarkB7 = DarkBlock(256, 2)
        self.transit_7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # (2, 2)
        self.GN_t7 = nn.GroupNorm(32, 256)

        self.DarkB8 = DarkBlock(256, 2)

        self.mish = Mish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = self.mish(self.GN1(self.conv1_1(image)))  # (N, 64, 512, 512)
        out = self.mish(self.GN2(self.conv1_2(out)))  # (N, 128, 512, 512)
        out = self.mish(self.GN3(self.conv1_3(out)))  # (N, 128, 256, 256)
        DB1_feats = self.DarkB1(out)  # (N, 128, 128, 128)
        DB2_feats = self.DarkB2(self.mish(self.GN_t1(self.transit_1(DB1_feats))))  # (N, 256, 128, 128)
        DB3_feats = self.DarkB3(self.mish(self.GN_t2(self.transit_2(DB2_feats))))  # (N, 512, 64, 64)
        DB4_feats = self.DarkB4(self.mish(self.GN_t3(self.transit_3(DB3_feats))))  # (N, 512, 32, 32)
        DB5_feats = self.DarkB5(self.mish(self.GN_t4(self.transit_4(DB4_feats))))  # (N, 256, 16, 16)
        DB6_feats = self.DarkB6(self.mish(self.GN_t5(self.transit_5(DB5_feats))))  # (N, 256, 8, 8)
        DB7_feats = self.DarkB7(self.mish(self.GN_t6(self.transit_6(DB6_feats))))  # (N, 192, 4, 4)
        DB8_feats = self.DarkB8(self.mish(self.GN_t7(self.transit_7(DB7_feats))))  # (N, 192, 2, 2)

        return DB2_feats, DB3_feats, DB4_feats, DB5_feats, DB6_feats, DB7_feats, DB8_feats


class NearestNeighborFusionModule(nn.Module):
    """
    Anchor Refinement Module:
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, feat_channels, internal_channels=256):
        """
        :param n_classes: number of different types of objects
        """
        super(NearestNeighborFusionModule, self).__init__()
        self.internal_channels = internal_channels
        # self.feat_channels = {'DB2': 256,
        #                       'DB3': 512,
        #                       'DB4': 512,
        #                       'DB5': 256,
        #                       'DB6': 256,
        #                       'DB7': 192,
        #                       'DB8': 192}
        self.feat_channels = feat_channels

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv = nn.Conv2d(self.feat_channels[0], self.internal_channels, kernel_size=1, padding=0)
        self.down_gn = nn.GroupNorm(32, self.internal_channels)

        # self.up_sample = nn.ConvTranspose2d(self.feat_channels[2], self.feat_channels[2], kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1, bias=self.use_bias)
        # self.up_sample_gn = nn.GroupNorm(32, self.feat_channels[2])
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = nn.Conv2d(self.feat_channels[2], self.internal_channels, kernel_size=1, padding=0)
        self.up_gn = nn.GroupNorm(32, self.internal_channels)

        self.current_conv = nn.Conv2d(self.feat_channels[1], self.internal_channels, kernel_size=1, padding=0)
        self.current_gn = nn.GroupNorm(32, self.internal_channels)
        self.mish = Mish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, prev_feats, current_feats, next_feats):
        prev_feats = self.mish(self.down_gn(self.down_conv(self.down_sample(prev_feats))))
        current_feats = self.mish(self.current_gn(self.current_conv(current_feats)))
        # next_feats = self.mish(self.up_gn(self.up_conv(self.mish(self.up_sample_gn(self.up_sample(next_feats))))))
        next_feats = self.mish(self.up_gn(self.up_conv(self.up_sample(next_feats))))

        return prev_feats + current_feats + next_feats


class NEModule(nn.Module):
    def __init__(self, internal_channels=256, up_factor=4):
        super(NEModule, self).__init__()
        self.internal_channels = internal_channels
        self.up_factor = up_factor
        self.up_sample = nn.Upsample(scale_factor=self.up_factor, mode='bilinear', align_corners=True)
        self.gate_conv = nn.Conv2d(self.internal_channels, 1, kernel_size=1, padding=0)
        self.gate_act = nn.Sigmoid()

        self.mish = Mish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, prev_feats, next_feats):
        gated_feats = self.gate_act(self.gate_conv(self.up_sample(next_feats)))  # sigmoid outputs
        es_feats = prev_feats * gated_feats
        pruned_feats = prev_feats - es_feats

        return pruned_feats, es_feats


class NTModule(nn.Module):
    def __init__(self, internal_channels=256, down_factor=4):
        super(NTModule, self).__init__()
        self.internal_channels = internal_channels
        self.down_factor = down_factor
        self.down_sample = nn.AvgPool2d(kernel_size=self.down_factor, stride=self.down_factor)
        self.conv = nn.Conv2d(self.internal_channels, self.internal_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(32, self.internal_channels)

        self.mish = Mish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, es_feats, next_feats):
        es_feats = self.mish(self.gn(self.conv(self.down_sample(es_feats))))

        return next_feats + es_feats


class NeighborErasingTransferModule(nn.Module):
    """
    Transfer Connection Block Architecture
    To link between the ARM and ODM, we introduce the TCBs to convert features of different layers from the ARM,
    into the form required by the ODM, so that the ODM can share features from the ARM.
    """

    def __init__(self, internal_channels=256):
        """
        :param lateral_channels: forward feature channels
        :param channels: pyramidal feature channels
        :param internal_channels: internal conv channels fix to 256
        :param is_batchnorm: adding batch norm
        """
        super(NeighborErasingTransferModule, self).__init__()
        self.internal_channels = internal_channels
        self.nem = NEModule(self.internal_channels)
        self.ntm = NTModule(self.internal_channels)

    def forward(self, lower_feats, higher_feats):
        pruned_feats, es_feats = self.nem(lower_feats, higher_feats)
        enhanced_feats = self.ntm(es_feats, higher_feats)

        return pruned_feats, enhanced_feats


class DetectorConvolutions(nn.Module):
    """
    Object Detection Module.
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, n_classes, internal_channels=256):
        """
        :param n_classes: number of different types of objects
        """
        super(DetectorConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'DB3': 3,
                   'DB4': 3,
                   'DB5': 3,
                   'DB6': 3,
                   'DB7': 3,
                   'DB8': 3}

        self.feat_channels = {'DB2': 256,
                              'DB3': 512,
                              'DB4': 512,
                              'DB5': 256,
                              'DB6': 256,
                              'DB7': 256,
                              'DB8': 256}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv3 = nn.Conv2d(internal_channels, n_boxes['DB3'] * 4, kernel_size=3, padding=1)
        self.loc_conv4 = nn.Conv2d(internal_channels, n_boxes['DB4'] * 4, kernel_size=3, padding=1)
        self.loc_conv5 = nn.Conv2d(internal_channels, n_boxes['DB5'] * 4, kernel_size=3, padding=1)
        self.loc_conv6 = nn.Conv2d(internal_channels, n_boxes['DB6'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(self.feat_channels['DB7'], n_boxes['DB7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8 = nn.Conv2d(self.feat_channels['DB8'], n_boxes['DB8'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv3 = nn.Conv2d(internal_channels, n_boxes['DB3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv4 = nn.Conv2d(internal_channels, n_boxes['DB4'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv5 = nn.Conv2d(internal_channels, n_boxes['DB5'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv6 = nn.Conv2d(internal_channels, n_boxes['DB6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(self.feat_channels['DB7'], n_boxes['DB7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8 = nn.Conv2d(self.feat_channels['DB8'], n_boxes['DB8'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight.data)
                if c.bias is not None:
                    nn.init.constant_(c.bias.data, 0)

    def forward(self, conv3, conv4, conv5, conv6, conv7, conv8):
        """
        Forward propagation.
        input from TCB modules
        :param conv8: a tensor of dimensions (N, 256, 2, 2)
        :param conv7: a tensor of dimensions (N, 256, 4, 4)
        :param conv3: a tensor of dimensions (N, 512, 64, 64)
        :param conv4: a tensor of dimensions (N, 512, 32, 32)
        :param conv5: a tensor of dimensions (N, 256, 16, 16)
        :param conv6: a tensor of dimensions (N, 256, 8, 8)
        """
        batch_size = conv3.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv3 = self.loc_conv3(conv3)  # (N, 512, 64, 64)
        l_conv3 = l_conv3.permute(0, 2, 3, 1).contiguous()
        l_conv3 = l_conv3.view(batch_size, -1, 4)

        l_conv4 = self.loc_conv4(conv4)  # (N, 512, 32, 32)
        l_conv4 = l_conv4.permute(0, 2, 3, 1).contiguous()
        l_conv4 = l_conv4.view(batch_size, -1, 4)

        l_conv5 = self.loc_conv5(conv5)  # (N, 256, 16, 16)
        l_conv5 = l_conv5.permute(0, 2, 3, 1).contiguous()
        l_conv5 = l_conv5.view(batch_size, -1, 4)

        l_conv6 = self.loc_conv6(conv6)  # (N, 256, 8, 8)
        l_conv6 = l_conv6.permute(0, 2, 3, 1).contiguous()
        l_conv6 = l_conv6.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7)  # (N, 256, 8, 8)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8 = self.loc_conv8(conv8)  # (N, 256, 8, 8)
        l_conv8 = l_conv8.permute(0, 2, 3, 1).contiguous()
        l_conv8 = l_conv8.view(batch_size, -1, 4)

        # # Predict classes in localization boxes
        c_conv3 = self.cl_conv3(conv3)
        c_conv3 = c_conv3.permute(0, 2, 3, 1).contiguous()
        c_conv3 = c_conv3.view(batch_size, -1, self.n_classes)

        c_conv4 = self.cl_conv4(conv4)
        c_conv4 = c_conv4.permute(0, 2, 3, 1).contiguous()
        c_conv4 = c_conv4.view(batch_size, -1, self.n_classes)

        c_conv5 = self.cl_conv5(conv5)
        c_conv5 = c_conv5.permute(0, 2, 3, 1).contiguous()
        c_conv5 = c_conv5.view(batch_size, -1, self.n_classes)

        c_conv6 = self.cl_conv6(conv6)
        c_conv6 = c_conv6.permute(0, 2, 3, 1).contiguous()
        c_conv6 = c_conv6.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8 = self.cl_conv8(conv8)
        c_conv8 = c_conv8.permute(0, 2, 3, 1).contiguous()
        c_conv8 = c_conv8.view(batch_size, -1, self.n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv3, l_conv4, l_conv5, l_conv6, l_conv7, l_conv8], dim=1).contiguous()
        classes_scores = torch.cat([c_conv3, c_conv4, c_conv5, c_conv6, c_conv7, c_conv8], dim=1).contiguous()

        return locs, classes_scores


class NETNetDetector(nn.Module):
    """
    The RefineDet512 network - encapsulates the base VGG network, auxiliary, ARM and ODM
    """

    def __init__(self, n_classes, config, prior=0.01):
        super(NETNetDetector, self).__init__()
        self.device = config.device
        self.n_classes = n_classes - 1
        self.prior = prior

        self.feat_channels = {'DB2': 256,
                              'DB3': 512,
                              'DB4': 512,
                              'DB5': 256,
                              'DB6': 256,
                              'DB7': 256,
                              'DB8': 256}

        self.base = DarknetBase()
        self.nnfm_1 = NearestNeighborFusionModule([self.feat_channels['DB2'], self.feat_channels['DB3'],
                                                   self.feat_channels['DB4']], 256)
        self.nnfm_2 = NearestNeighborFusionModule([self.feat_channels['DB3'], self.feat_channels['DB4'],
                                                   self.feat_channels['DB5']], 256)
        self.nnfm_3 = NearestNeighborFusionModule([self.feat_channels['DB4'], self.feat_channels['DB5'],
                                                   self.feat_channels['DB6']], 256)
        self.nnfm_4 = NearestNeighborFusionModule([self.feat_channels['DB5'], self.feat_channels['DB6'],
                                                   self.feat_channels['DB7']], 256)
        self.netm_1 = NeighborErasingTransferModule(256)
        self.netm_2 = NeighborErasingTransferModule(256)

        self.detect_convs = DetectorConvolutions(self.n_classes, 256)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

        # initialization for focal loss
        self.detect_convs.cl_conv3.weight.data.fill_(0)
        self.detect_convs.cl_conv3.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.detect_convs.cl_conv4.weight.data.fill_(0)
        self.detect_convs.cl_conv4.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.detect_convs.cl_conv5.weight.data.fill_(0)
        self.detect_convs.cl_conv5.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.detect_convs.cl_conv6.weight.data.fill_(0)
        self.detect_convs.cl_conv6.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.detect_convs.cl_conv7.weight.data.fill_(0)
        self.detect_convs.cl_conv7.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.detect_convs.cl_conv8.weight.data.fill_(0)
        self.detect_convs.cl_conv8.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: 22536 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        p_0, p_1, p_2, p_3, p_4, p_5, p_6 = self.base(image)
        nn_feat1 = self.nnfm_1(p_0, p_1, p_2)
        nn_feat2 = self.nnfm_2(p_1, p_2, p_3)
        nn_feat3 = self.nnfm_3(p_2, p_3, p_4)
        nn_feat4 = self.nnfm_4(p_3, p_4, p_5)

        dh1, dh3 = self.netm_1(nn_feat1, nn_feat3)
        dh2, dh4 = self.netm_2(nn_feat2, nn_feat4)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, scores = self.detect_convs(dh1, dh2, dh3, dh4, p_5, p_6)

        return locs, scores

    def create_prior_boxes(self):
        """
        Create the 22536 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (22536, 4)
        """
        fmap_dims = {'DB3': [64, 64],
                     'DB4': [32, 32],
                     'DB5': [16, 16],
                     'DB6': [8, 8],
                     'DB7': [4, 4],
                     'DB8': [2, 2]}

        obj_scales = {'DB3': 0.035,
                      'DB4': 0.07,
                      'DB5': 0.14,
                      'DB6': 0.28,
                      'DB7': 0.56,
                      'DB8': 0.84}
        scale_factor = [1.]
        # scale_factor = [2. ** 0, 2. ** (1 / 3.), 2. ** (2 / 3.)]
        aspect_ratios = {'DB3': [1., 2., 0.5],
                         'DB4': [1., 2., 0.5],
                         'DB5': [1., 2., 0.5],
                         'DB6': [1., 2., 0.5],
                         'DB7': [1., 2., 0.5],
                         'DB8': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap][0]):
                for j in range(fmap_dims[fmap][1]):
                    cx = (j + 0.5) / fmap_dims[fmap][1]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap][0]

                    for ratio in aspect_ratios[fmap]:
                        for fac in scale_factor:
                            prior_boxes.append([cx, cy, obj_scales[fmap] * fac * sqrt(ratio),
                                                obj_scales[fmap] * fac / sqrt(ratio)])

        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device).contiguous()
        prior_boxes.clamp_(0, 1)

        return prior_boxes


class NETNetDetectorLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, threshold=0.7, theta=0.1):
        super(NETNetDetectorLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes - 1

        self.theta = theta

        self.regression_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Ciou')
        # self.regression_loss = SmoothL1Loss()
        # self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.classification_loss = SigmoidFocalLoss(gamma=2.0, alpha=0.25, config=config)

    def forward(self, odm_locs, odm_scores, boxes, labels):
        """
        :param odm_locs: predicted bboxes
        :param odm_scores: predicted scores for each bbox
        :param boxes: gt
        :param labels: gt
        :return:
        """
        batch_size = odm_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = odm_scores.size(2)

        assert n_priors == odm_locs.size(1) == odm_scores.size(1)

        decoded_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        # true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        # For each image
        for i in range(batch_size):
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # initial overlap

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (22536)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            overlap_for_each_object, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            prior_for_each_object = prior_for_each_object[overlap_for_each_object > 0]
            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            if len(prior_for_each_object) > 0:
                overlap_for_each_prior.index_fill_(0, prior_for_each_object, 1.0)

            for j in range(prior_for_each_object.size(0)):
                object_for_each_prior[prior_for_each_object[j]] = j

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = -1
            label_for_each_prior[overlap_for_each_prior < self.threshold - 0.25] = 0
            # label in 0.45 - 0.7 is not used

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            # true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]),
            #                                       xy_to_cxcy(decoded_arm_locs[i]))
            # true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
            true_locs[i] = boxes[i][object_for_each_prior]
            decoded_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(odm_locs[i], self.priors_cxcy))

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0

        # LOCALIZATION LOSS
        loc_loss = self.regression_loss(decoded_locs[positive_priors].view(-1, 4),
                                        true_locs[positive_priors].view(-1, 4))
        # loc_loss = self.regression_loss(odm_locs[positive_priors].view(-1, 4),
        #                                 true_locs_encoded[positive_priors].view(-1, 4))

        # CONFIDENCE LOSS
        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum().float()  # (N)
        conf_loss = self.classification_loss(odm_scores.view(-1, n_classes),
                                    true_classes.view(-1)) / n_positives
        # n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)
        #
        # # First, find the loss for all priors
        # conf_loss_all = self.cross_entropy(odm_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        # # conf_loss_all = self.CELoss(odm_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        # conf_loss_all = conf_loss_all.view(batch_size, -1)  # (N, 8732)
        #
        # # We already know which priors are positive
        # conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))
        #
        # # Next, find which priors are hard-negative
        # # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        # conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        # conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        #
        # conf_loss_neg, _ = conf_loss_neg.sort(dim=-1,
        #                                       descending=True)  # (N, 8732), sorted by decreasing hardness
        # hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
        #     self.device)  # (N, 8732)
        # hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        # conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))
        #
        # conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
