from torch import nn
import torch.nn.functional as F
from math import sqrt
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SmoothL1Loss, LabelSmoothingLoss, SigmoidFocalLoss, focal_loss
from metrics import find_jaccard_overlap
from operators.iou_utils import find_distance
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

        # print('feature maps:')
        # print([p.size() for p in [DB2_feats, DB3_feats, DB4_feats, DB5_feats, DB6_feats, DB7_feats, DB8_feats]])

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
        n_boxes = {'DB3': 1,
                   'DB4': 1,
                   'DB5': 1,
                   'DB6': 1,
                   'DB7': 1,
                   'DB8': 1}

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
        # print([p.size() for p in [l_conv3, l_conv4, l_conv5, l_conv6, l_conv7, l_conv8]])
        # print([p.size() for p in [c_conv3, c_conv4, c_conv5, c_conv6, c_conv7, c_conv8]])
        # exit()
        # return [l_conv3, l_conv4, l_conv5, l_conv6, l_conv7, l_conv8], \
        #        [c_conv3, c_conv4, c_conv5, c_conv6, c_conv7, c_conv8]


class ATSSNETNetDetector(nn.Module):
    """
    The RefineDet512 network - encapsulates the base VGG network, auxiliary, ARM and ODM
    """

    def __init__(self, n_classes, config, prior=0.01):
        super(ATSSNETNetDetector, self).__init__()
        self.device = config.device
        self.n_classes = n_classes
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

        # self.detect_convs.loc_conv3.weight.data.fill_(0)
        # self.detect_convs.loc_conv3.bias.data.fill_(0)
        # self.detect_convs.loc_conv4.weight.data.fill_(0)
        # self.detect_convs.loc_conv4.bias.data.fill_(0)
        # self.detect_convs.loc_conv5.weight.data.fill_(0)
        # self.detect_convs.loc_conv5.bias.data.fill_(0)
        # self.detect_convs.loc_conv6.weight.data.fill_(0)
        # self.detect_convs.loc_conv6.bias.data.fill_(0)
        # self.detect_convs.loc_conv7.weight.data.fill_(0)
        # self.detect_convs.loc_conv7.bias.data.fill_(0)
        # self.detect_convs.loc_conv8.weight.data.fill_(0)
        # self.detect_convs.loc_conv8.bias.data.fill_(0)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

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

        # print('detection feature maps:')
        # print([p.size() for p in [dh1, dh2, dh3, dh4, p_5, p_6]])
        # exit()

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

        obj_scales = {'DB3': 0.03,
                      'DB4': 0.07,
                      'DB5': 0.15,
                      'DB6': 0.3,
                      'DB7': 0.55,
                      'DB8': 0.75}
        scale_factor = [1.]
        # scale_factor = [2. ** 0, 2. ** (1 / 3.), 2. ** (2 / 3.)]
        # aspect_ratios = {'DB3': [1., 2., 0.5],
        #                  'DB4': [1., 2., 0.5],
        #                  'DB5': [1., 2., 0.5],
        #                  'DB6': [1., 2., 0.5],
        #                  'DB7': [1., 2., 0.5],
        #                  'DB8': [1., 2., 0.5]}
        aspect_ratios = {'DB3': [1.],
                         'DB4': [1.],
                         'DB5': [1.],
                         'DB6': [1.],
                         'DB7': [1.],
                         'DB8': [1.]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            temp_prior_boxes = []
            for i in range(fmap_dims[fmap][0]):
                for j in range(fmap_dims[fmap][1]):
                    cx = (j + 0.5) / fmap_dims[fmap][1]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap][0]

                    for ratio in aspect_ratios[fmap]:
                        for fac in scale_factor:
                            temp_prior_boxes.append([cx, cy, obj_scales[fmap] * fac * sqrt(ratio),
                                                     obj_scales[fmap] * fac / sqrt(ratio)])

            temp_prior_boxes = torch.FloatTensor(temp_prior_boxes).to(self.device).contiguous()
            temp_prior_boxes.clamp_(0, 1)
            prior_boxes.append(temp_prior_boxes)

        return prior_boxes


class ATSSNETNetDetectorLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, n_candidates=9):
        super(ATSSNETNetDetectorLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = [cxcy_to_xy(prior) for prior in self.priors_cxcy]
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.n_candidates = n_candidates

        self.prior_split_points = [0, 4096, 5120, 5376, 5440, 5456, 5460]

        self.regression_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        # self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        # self.FocalLoss = SigmoidFocalLoss(gamma=2.0, alpha=0.25, config=config)
        self.FocalLoss = focal_loss

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        :param predicted_scores: list of predicted class scores for each feature level
        :param predicted_locs: list of predicted bboxes for each feature level
        :param boxes: gt
        :param labels: gt
        :return:
        """
        n_levels = len(self.priors_cxcy)
        batch_size = predicted_locs.size(0)
        n_priors = [prior.size(0) for prior in self.priors_cxcy]
        n_classes = predicted_scores.size(2)

        # split the prediction according to the levels
        # split_predicted_locs = []
        # split_predicted_scores = []
        # for s in range(len(self.priors_cxcy)):
        #     split_predicted_locs.append(predicted_locs[self.prior_split_points[s]:self.prior_split_points[s + 1], :])
        #     split_predicted_scores.append(predicted_scores[self.prior_split_points[s]:self.prior_split_points[s + 1], :])
        #
        # predicted_locs = split_predicted_locs
        # predicted_scores = split_predicted_scores
        #
        # print('predicted_locs:', [p.size() for p in predicted_locs])

        # total_predicted_bboxes = sum([loc.size(1) for loc in predicted_locs])
        # print(total_predicted_bboxes, n_priors, n_classes)
        # assert sum(n_priors) == total_predicted_bboxes

        # decoded_odm_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        # true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        # true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        # true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)
        decoded_locs = list()  # length is the batch size
        true_locs = list()
        true_classes = list()
        # positive_priors = list()
        predicted_class_scores = list()

        # For each image
        for i in range(batch_size):
            image_bboxes = boxes[i]
            batch_split_predicted_locs = []
            batch_split_predicted_scores = []
            for s in range(len(self.priors_cxcy)):
                batch_split_predicted_locs.append(
                    predicted_locs[i][self.prior_split_points[s]:self.prior_split_points[s + 1], :])
                batch_split_predicted_scores.append(
                    predicted_scores[i][self.prior_split_points[s]:self.prior_split_points[s + 1], :])

            # print('Lenght of priors:', len(self.priors_cxcy))
            # print('Original shape', predicted_locs[i].size())
            # print([p.size() for p in batch_split_predicted_locs])
            # exit()
            # positive_samples = list()
            # predicted_pos_locs = list()
            positive_samples_idx = list()
            positive_overlaps = list()
            overlap = list()
            for level in range(n_levels):
                distance = find_distance(xy_to_cxcy(image_bboxes), self.priors_cxcy[level])  # n_bboxes, n_priors
                _, top_idx_level = torch.topk(-1. * distance, min(self.n_candidates, distance.size(1)), dim=1)
                # print(distance.size(), top_idx_level.size())
                # print(top_idx_level)

                # level_priors = self.priors_cxcy[level].unsqueeze(0).expand(image_bboxes.size(0),
                #                                                            self.priors_cxcy[level].size(0), 4)
                # print('level_priors: ', level_priors.size())
                # level_predictions = predicted_locs[i][level].unsqueeze(0).expand(image_bboxes.size(0),
                #                                             self.priors_cxcy[level].size(0))
                # positive_samples.append(level_priors[top_idx_level])
                positive_samples_idx.append(top_idx_level)
                # predicted_pos_locs.append(cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[level]), self.priors_cxcy[level]))
                overlap_level = find_jaccard_overlap(image_bboxes, self.priors_xy[level])  # overlap for each level
                positive_overlaps.append(torch.gather(overlap_level, dim=1, index=top_idx_level))
                overlap.append(overlap_level)

            positive_overlaps_cat = torch.cat(positive_overlaps, dim=1)  # n_bboxes, n_priors * 6level
            # print('positive_overlaps_cat shape: ', positive_overlaps_cat.size())
            overlap_mean = torch.mean(positive_overlaps_cat, dim=1)
            overlap_std = torch.std(positive_overlaps_cat, dim=1)
            # print(overlap_mean, overlap_std)
            iou_threshold = overlap_mean + overlap_std  # n_bboxes, for each object, we have one threshold
            # print([p.size() for p in positive_overlaps])
            # print('threshold: ', iou_threshold)
            # print('\n')

            # one prior can only be associated to one gt object
            # For each prior, find the object that has the maximum overlap, return [value, indices]
            # overlap = torch.cat(overlap, dim=1)
            true_classes_level = list()
            true_locs_level = list()
            decoded_locs_level = list()
            for level in range(n_levels):
                positive_priors_per_level = torch.zeros((self.priors_cxcy[level].size(0)),
                                                        dtype=torch.uint8).to(self.device)  # indexing, (n,) shape
                label_for_each_prior_per_level = torch.zeros((self.priors_cxcy[level].size(0)),
                                                             dtype=torch.long).to(self.device)
                true_locs_per_level = list()
                decoded_locs_per_level = list()
                total_decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(batch_split_predicted_locs[level], self.priors_cxcy[level]))
                # print(positive_priors_per_level.size(), label_for_each_prior_per_level.size())
                # print(level, 'total decoded priors shape: ', total_decoded_locs.size())
                # overlap_for_each_prior, object_for_each_prior = overlap[level].max(dim=0)
                # overlap_for_each_object, prior_for_each_object = overlap[level].max(dim=1)
                for ob in range(image_bboxes.size(0)):
                    for c in range(len(positive_samples_idx[level][ob])):
                        # print(ob, c, 'Range for c: ', len(positive_samples_idx[level][ob]))
                        current_iou = positive_overlaps[level][ob, c]
                        current_bbox = image_bboxes[ob, :]
                        current_prior = self.priors_cxcy[level][positive_samples_idx[level][ob, c], :]
                        # print(current_iou, iou_threshold[ob], current_bbox, current_prior)

                        if current_iou > iou_threshold[ob]:
                            if current_bbox[0] <= current_prior[0] <= current_bbox[2] \
                                    and current_bbox[1] <= current_prior[1] <= current_bbox[3]:
                                # print('------------------------------------------------------------------')
                                positive_priors_per_level[positive_samples_idx[level][ob, c]] = 1
                                # if current_iou == overlap_for_each_prior[positive_samples_idx[level][ob, c]]:
                                # print(label_for_each_prior_per_level[positive_samples_idx[level][ob, c]])
                                label_for_each_prior_per_level[positive_samples_idx[level][ob, c]] = labels[i][ob]
                                # print(label_for_each_prior_per_level[positive_samples_idx[level][ob, c]])
                                # print('Details:')
                                # print(positive_samples_idx[level][ob, c], labels[i][ob])
                                # exit()

                                temp_true_locs = image_bboxes[ob, :].unsqueeze(0)  # (1, 4)
                                temp_decoded_locs = total_decoded_locs[positive_samples_idx[level][ob, c], :].unsqueeze(
                                    0)  # (1, 4)
                                true_locs_per_level.append(temp_true_locs)
                                decoded_locs_per_level.append(temp_decoded_locs)
                                # print(temp_true_locs.size(), temp_decoded_locs.size())
                                # exit()
                # print(label_for_each_prior_per_level.size(), len(self.priors_cxcy[level]))
                # assert label_for_each_prior_per_level.size(0) == self.priors_cxcy[level].size(0)
                if len(true_locs_per_level) > 0:
                    true_locs_level.append(torch.cat(true_locs_per_level, dim=0).view(-1, 4))  # (1, n_l * 4)
                    decoded_locs_level.append(torch.cat(decoded_locs_per_level, dim=0).view(-1, 4))
                    assert torch.cat(decoded_locs_per_level, dim=0).view(-1, 4).size(0) == torch.cat(
                        true_locs_per_level,
                        dim=0).view(-1, 4).size(0)
                true_classes_level.append(label_for_each_prior_per_level)
                assert len(label_for_each_prior_per_level) == len(batch_split_predicted_locs[level])

            # Store
            true_classes.append(torch.cat(true_classes_level, dim=0))  # batch_size, n_priors
            predicted_class_scores.append(torch.cat(batch_split_predicted_scores, dim=0))
            if len(true_locs_level) > 0:
                true_locs.append(torch.cat(true_locs_level, dim=0))  # batch_size, n_pos, 4
                # print(odm_locs.size(), decoded_arm_locs.size())
                decoded_locs.append(torch.cat(decoded_locs_level, dim=0))

        # assemble all samples from batches
        true_classes = torch.cat(true_classes, dim=0)
        positive_priors = true_classes > 0
        predicted_scores = torch.cat(predicted_class_scores, dim=0)
        true_locs = torch.cat(true_locs, dim=0)
        decoded_locs = torch.cat(decoded_locs, dim=0)

        # print('Final stored values:')
        # print(true_locs.size(), true_classes.size())
        # print(decoded_locs.size(), predicted_scores.size())
        # print('true locs:', true_locs[:15, :])
        # print('true classes:', true_classes[4000:])
        # print(decoded_locs[:15, :])
        #
        # exit()

        # LOCALIZATION LOSS
        loc_loss = self.regression_loss(decoded_locs, true_locs)

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum().float()

        # First, find the loss for all priors
        conf_loss = self.FocalLoss(predicted_scores, true_classes, device=self.device) / true_classes.size(0) * 1.
        # conf_loss = self.FocalLoss(predicted_scores, true_classes) / n_positives

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
