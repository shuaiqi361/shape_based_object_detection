from torch import nn
import torch.nn.functional as F
from math import sqrt
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SmoothL1Loss, LabelSmoothingLoss
from metrics import find_jaccard_overlap
from .modules import Mish, AdaptivePooling, Residual, ConvBNAct, DualAdaptivePooling, ConvGNAct, Upsample


class DarkBlock(nn.Module):
    def __init__(self, in_planes, n_blocks):
        super(DarkBlock, self).__init__()
        self.in_planes = in_planes  # 64
        # self.out_planes = out_planes  # 128
        self.n_blocks = n_blocks
        self.Block = self.__make_block__(self.n_blocks)

    def __make_block__(self, n_blocks):
        block = []
        for i in range(n_blocks):
            layer = self.__make_layers__(ConvBNAct, Residual)
            block += layer

        return nn.Sequential(*block)

    def __make_layers__(self, conv_block, residual_block):
        layers = list()
        layers.append(conv_block(self.in_planes, self.in_planes // 2, kernel_size=1, padding=0))
        layers.append(conv_block(self.in_planes // 2, self.in_planes, kernel_size=3, padding=1))
        layers.append(residual_block(self.in_planes))

        return layers

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

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.GN1 = nn.GroupNorm(num_groups=16, num_channels=32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)  # (512+, 512+)
        self.GN2 = nn.GroupNorm(16, 64)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # (256, 256)
        self.GN3 = nn.GroupNorm(16, 64)

        self.DarkB1 = DarkBlock(64, 1)
        self.transit_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.GN_t1 = nn.GroupNorm(32, 128)
        self.adap_pool = DualAdaptivePooling(128, 128, adaptive_size=(56 * 2, 96 * 2), use_gn=True)  # (128, 128)

        self.DarkB2 = DarkBlock(128, 3)  # (56, 96)
        self.transit_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.GN_t2 = nn.GroupNorm(32, 256)

        self.DarkB3 = DarkBlock(256, 2)  # (28, 48)
        self.transit_3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.GN_t3 = nn.GroupNorm(32, 512)

        self.DarkB4 = DarkBlock(512, 2)
        # self.transit_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=2)
        # self.GN_t4 = nn.GroupNorm(32, 256)
        #
        # self.DarkB5 = DarkBlock(256, 2)  # (8, 8)
        # self.transit_5 = nn.Conv2d(256, 192, kernel_size=3, padding=1, stride=2)
        # self.GN_t5 = nn.GroupNorm(32, 192)
        #
        # self.DarkB6 = DarkBlock(192, 2)
        self.mish = Mish()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)
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
        DB1_feats = self.DarkB1(out)  # (N, 256, _, _)
        DB2_feats = self.DarkB2(
            self.adap_pool(self.mish(self.GN_t1(self.transit_1(DB1_feats)))))  # (N, 256, 56 * 2, 96 * 2)
        DB3_feats = self.DarkB3(self.mish(self.GN_t2(self.transit_2(DB2_feats))))  # (N, 256, 56, 96)
        DB4_feats = self.DarkB4(self.mish(self.GN_t3(self.transit_3(DB3_feats))))  # (N, 512, 28, 48)
        # DB5_feats = self.DarkB5(self.mish(self.GN_t4(self.transit_4(DB4_feats))))  # (N, 512, 14, 24)
        # DB6_feats = self.DarkB6(self.mish(self.GN_t5(self.transit_5(DB5_feats))))  # (N, 256, 7, 12)

        return DB2_feats, DB3_feats, DB4_feats  # , DB5_feats, DB6_feats


class AuxilliaryConvolution(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps
    Feel free to substitute with other pre-trained backbones
    """

    def __init__(self):
        super(AuxilliaryConvolution, self).__init__()

        # self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # self.GN1 = nn.GroupNorm(num_groups=16, num_channels=32)
        # self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)  # (512+, 512+)
        # self.GN2 = nn.GroupNorm(16, 64)
        # self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # (256, 256)
        # self.GN3 = nn.GroupNorm(16, 64)
        #
        # self.DarkB1 = DarkBlock(64, 1)
        # self.transit_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        # self.GN_t1 = nn.GroupNorm(32, 128)
        # self.adap_pool = DualAdaptivePooling(128, 128, adaptive_size=(56 * 2, 96 * 2), use_gn=True)  # (128, 128)

        # self.DarkB2 = DarkBlock(128, 3)  # (56, 96)
        # self.transit_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        # self.GN_t2 = nn.GroupNorm(32, 256)
        #
        # self.DarkB3 = DarkBlock(256, 2)  # (28, 48)
        self.transit_3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.GN_t3 = nn.GroupNorm(32, 512)

        self.DarkB4 = DarkBlock(512, 2)  # (16, 16)
        self.transit_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=2)
        self.GN_t4 = nn.GroupNorm(32, 256)

        self.DarkB5 = DarkBlock(256, 2)  # (8, 8)
        self.transit_5 = nn.Conv2d(256, 192, kernel_size=3, padding=1, stride=2)
        self.GN_t5 = nn.GroupNorm(32, 192)

        self.DarkB6 = DarkBlock(192, 2)
        self.mish = Mish()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, db3_feats):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: lower-level feature maps conv4_3 and conv7
        """
        # DB3_feats = self.DarkB3(self.mish(self.GN_t2(self.transit_2(db3_feats))))  # (N, 256, 56, 96)
        DB4_feats = self.DarkB4(self.mish(self.GN_t3(self.transit_3(db3_feats))))  # (N, 512, 28, 48)
        DB5_feats = self.DarkB5(self.mish(self.GN_t4(self.transit_4(DB4_feats))))  # (N, 512, 14, 24)
        DB6_feats = self.DarkB6(self.mish(self.GN_t5(self.transit_5(DB5_feats))))  # (N, 256, 7, 12)

        return DB4_feats, DB5_feats, DB6_feats


class AttentionSegConvolution(nn.Module):
    """
    Generate the segmentation map as the self-attention mechanism
    """

    def __init__(self):
        super(AttentionSegConvolution, self).__init__()
        self.feat_channels = {'DB2': 128,
                              'DB3': 256,
                              'DB4': 512,
                              'DB5': 256,
                              'DB6': 192}

        self.Up_conv1 = Upsample(512, 256)
        self.Up_conv2 = Upsample(256, 128)
        self.transit_1 = ConvGNAct(self.feat_channels['DB2'], self.feat_channels['DB2'], kernel_size=3, padding=1)
        self.transit_2 = ConvGNAct(self.feat_channels['DB2'], self.feat_channels['DB2'], kernel_size=3, padding=1)
        self.out = nn.Conv2d(self.feat_channels['DB2'], 1, kernel_size=1, padding=0)
        self.out_act = nn.Sigmoid()

    def forward(self, db2_feats, db3_feats, db4_feats):
        up_4 = self.Up_conv1(db4_feats)  # (N, 256, 56, 96)
        up_3 = self.Up_conv2(up_4 + db3_feats)  # (N, 256, 56 * 2, 96 * 2)
        out = self.transit_1(up_3 + db2_feats)
        out = self.transit_2(out)
        out = self.out_act(self.out(out))

        return out


class TCBConvolutions(nn.Module):
    """
    Anchor Refinement Module:
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, internal_channels=192):
        """
        :param n_classes: number of different types of objects
        """
        super(TCBConvolutions, self).__init__()

        self.feat_channels = {'DB3': 256,
                              'DB4': 512,
                              'DB5': 256,
                              'DB6': 192}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.tcb_conv4_3 = TCB(self.feat_channels['DB3'], self.feat_channels['DB4'], internal_channels)
        self.tcb_conv7 = TCB(self.feat_channels['DB4'], self.feat_channels['DB5'], internal_channels)
        self.tcb_conv8_2 = TCB(self.feat_channels['DB5'], self.feat_channels['DB6'], internal_channels)
        self.tcb_conv9_2 = TCBTail(self.feat_channels['DB6'], internal_channels)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats):
        """
        Forward propagation. To ge initial offsets w.r.t. anchors anc binary labels

        """
        # batch_size = conv4_3_feats.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.tcb_conv4_3(conv4_3_feats, conv7_feats)  # (N, 256, 64, 64)
        l_conv7 = self.tcb_conv7(conv7_feats, conv8_2_feats)  # (N, 256, 32, 32)
        l_conv8_2 = self.tcb_conv8_2(conv8_2_feats, conv9_2_feats)  # (N, 256, 16, 16)
        l_conv9_2 = self.tcb_conv9_2(conv9_2_feats)  # (N, 256, 8, 8)

        return l_conv4_3, l_conv7, l_conv8_2, l_conv9_2


class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    To link between the ARM and ODM, we introduce the TCBs to convert features of different layers from the ARM,
    into the form required by the ODM, so that the ODM can share features from the ARM.
    """

    def __init__(self, lateral_channels, channels, internal_channels=192, is_groupnorm=True):
        """
        :param lateral_channels: forward feature channels
        :param channels: pyramidal feature channels
        :param internal_channels: internal conv channels fix to 256
        :param is_batchnorm: adding batch norm
        """
        super(TCB, self).__init__()
        self.is_groupnorm = is_groupnorm
        self.use_bias = not self.is_groupnorm
        self.out_channels = internal_channels

        self.conv1 = nn.Conv2d(lateral_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv3 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)

        self.deconv = nn.ConvTranspose2d(channels, internal_channels, kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=self.use_bias)

        self.mish = Mish()

        if self.is_groupnorm:
            self.gn1 = nn.GroupNorm(32, internal_channels)
            self.gn2 = nn.GroupNorm(32, internal_channels)
            self.deconv_gn = nn.GroupNorm(32, internal_channels)
            self.gn3 = nn.GroupNorm(32, internal_channels)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, lateral, higher_level):
        if self.is_groupnorm:
            lateral_out = self.mish(self.gn1(self.conv1(lateral)))
            out = self.mish(self.gn2(self.conv2(lateral_out)) + self.deconv_gn(self.deconv(higher_level)))
            out = self.mish(self.gn3(self.conv3(out)))
        else:
            lateral_out = self.mish(self.conv1(lateral))
            out = self.mish(self.conv2(lateral_out) + self.deconv(higher_level))
            out = self.mish(self.conv3(out))

        return out


class TCBTail(nn.Module):
    """
    Transfer Connection Block Architecture
    To link between the ARM and ODM, we introduce the TCBs to convert features of different layers from the ARM,
    into the form required by the ODM, so that the ODM can share features from the ARM.
    """

    def __init__(self, lateral_channels, internal_channels=192, is_groupnorm=True):
        """
        :param lateral_channels: forward feature channels
        :param channels: pyramidal feature channels
        :param internal_channels: internal conv channels fix to 192
        :param is_batchnorm: adding batch norm
        """
        super(TCBTail, self).__init__()
        self.is_groupnorm = is_groupnorm
        self.use_bias = not self.is_groupnorm
        self.out_channels = internal_channels

        self.conv1 = nn.Conv2d(lateral_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv3 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)

        self.mish = Mish()

        if self.is_groupnorm:
            self.gn1 = nn.GroupNorm(32, internal_channels)
            self.gn2 = nn.GroupNorm(32, internal_channels)
            self.gn3 = nn.GroupNorm(32, internal_channels)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, lateral):
        if self.is_groupnorm:
            lateral_out = self.mish(self.gn1(self.conv1(lateral)))
            out = self.mish(self.gn2(self.conv2(lateral_out)))
            out = self.mish(self.gn3(self.conv3(out)))
        else:
            lateral_out = self.mish(self.conv1(lateral))
            out = self.mish(self.conv2(lateral_out))
            out = self.mish(self.conv3(out))

        return out


class ODMConvolutions(nn.Module):
    """
    Object Detection Module.
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, n_classes, internal_channels=192):
        """
        :param n_classes: number of different types of objects
        """
        super(ODMConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'DB3': 2,
                   'DB4': 4,
                   'DB5': 5,
                   'DB6': 5}

        self.feat_channels = {'DB3': 256,
                              'DB4': 512,
                              'DB5': 256,
                              'DB6': 192}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(internal_channels, n_boxes['DB3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(internal_channels, n_boxes['DB4'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(internal_channels, n_boxes['DB5'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(internal_channels, n_boxes['DB6'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(internal_channels, n_boxes['DB3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(internal_channels, n_boxes['DB4'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(internal_channels, n_boxes['DB5'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(internal_channels, n_boxes['DB6'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                # n = c.kernel_size[0] * c.kernel_size[1] * c.out_channels
                # c.weight.data.normal_(0, sqrt(2. / n))
                nn.init.xavier_normal_(c.weight.data)
                # nn.init.kaiming_normal_(c.weight.data)
                if c.bias is not None:
                    nn.init.constant_(c.bias.data, 0)

    def forward(self, conv4_3_tcb, conv7_tcb, conv8_2_tcb, conv9_2_tcb):
        """
        Forward propagation.
        input from TCB modules
        :param conv9_2_tcb:
        :param conv8_2_tcb:
        :param conv7_tcb:
        :param conv4_3_tcb:
        :return: 16320 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_tcb.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_tcb)  # (N, 16, 64, 64)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 64, 64, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 16384, 4), there are a total 16384 boxes

        l_conv7 = self.loc_conv7(conv7_tcb)  # (N, 24, 16, 16)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 16, 16, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 1536, 4), there are a total 1536 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_tcb)  # (N, 24, 8, 8)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 8, 8, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 384, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_tcb)  # (N, 24, 4, 4)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 4, 4, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 96, 4)

        # # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_tcb)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)

        c_conv7 = self.cl_conv7(conv7_tcb)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_tcb)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_tcb)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        # A total of 16320 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2], dim=1).contiguous()
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2], dim=1).contiguous()

        return locs, classes_scores


class DarkTrafficAttentionDetector(nn.Module):
    """
    The RefineDet512 network - encapsulates the base VGG network, auxiliary, ARM and ODM
    """

    def __init__(self, n_classes, config):
        super(DarkTrafficAttentionDetector, self).__init__()
        self.device = config.device
        self.n_classes = n_classes
        self.base = DarknetBase()
        self.theta = 0.01

        self.seg_convs = AttentionSegConvolution()
        self.odm_convs = ODMConvolutions(self.n_classes)
        self.tcb_convs = TCBConvolutions()
        self.aux_convs = AuxilliaryConvolution()

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: 22536 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv2_feats, conv3_feats, conv4_feats = self.base(image)
        seg_out = self.seg_convs(conv2_feats, conv3_feats, conv4_feats)
        attention_map = F.interpolate(seg_out, scale_factor=2)  # (N, 1, 56, 96)
        conv3_feats = conv3_feats * attention_map
        aux4_feats, aux5_feats, aux6_feats = self.aux_convs(conv3_feats)

        tcb_conv4_3, tcb_conv7, tcb_conv8_2, tcb_conv9_2 = \
            self.tcb_convs(conv3_feats, aux4_feats, aux5_feats, aux6_feats)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        odm_locs, odm_scores = self.odm_convs(tcb_conv4_3, tcb_conv7, tcb_conv8_2, tcb_conv9_2)

        return odm_locs, odm_scores, attention_map

    def offset2bbox(self, arm_locs, odm_locs):
        batch_size = arm_locs.size(0)
        n_priors = self.priors_cxcy.size(0)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)

        for i in range(batch_size):
            init_bbox_cxcy = gcxgcy_to_cxcy(arm_locs[i], self.priors_cxcy)
            true_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(odm_locs[i], init_bbox_cxcy))

        return true_locs

    def create_prior_boxes(self):
        """
        Create the 22536 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (22536, 4)
        """
        fmap_dims = {'DB3': [56, 96],
                     'DB4': [28, 48],
                     'DB5': [14, 24],
                     'DB6': [7, 12]}

        obj_scales = {'DB3': 0.05,
                      'DB4': 0.12,
                      'DB5': 0.3,
                      'DB6': 0.55}
        scale_factor = [1.]
        # scale_factor = [2. ** 0, 2. ** (1 / 3.), 2. ** (2 / 3.)]
        aspect_ratios = {'DB3': [1.],
                         'DB4': [1., 2., 0.5],
                         'DB5': [1., 2., 3., 0.5],
                         'DB6': [1., 2., 3., 0.5]}

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

                        if ratio == 1.:
                            try:
                                # use the geometric mean to calculate the additional scale for each level of fmap
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last object scale, there is no "next" scale
                            except IndexError:
                                additional_scale = 0.7
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device).contiguous()
        prior_boxes.clamp_(0, 1)

        return prior_boxes


class DarkTrafficAttentionDetectorLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, threshold=0.4, neg_pos_ratio=2, theta=0.1):
        super(DarkTrafficAttentionDetectorLoss, self).__init__()
        self.fmap_dims = {'DB3': [56, 96],
                          'DB4': [28, 48],
                          'DB5': [14, 24],
                          'DB6': [7, 12]}
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.config = config
        self.theta = theta  # threshold for overlap between ignored regions and priors

        # self.arm_loss = SmoothL1Loss(reduction='mean')
        # self.odm_loss = SmoothL1Loss(reduction='mean')
        # self.Diou_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        # self.arm_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        self.odm_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        # self.arm_cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.odm_cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.seg_loss = nn.BCELoss(reduction='sum')

    def compute_odm_loss(self, odm_locs, odm_scores, boxes, labels, ignored_regions):
        """
        :param ignored_regions: regions that have no annotations, these regions should not be used for negative examples
        :param arm_locs: serve as "anchor boxes"
        :param arm_scores:
        :param odm_locs:
        :param odm_scores:
        :param boxes:
        :param labels:
        :return:
        """
        # print(arm_scores.size(), arm_locs.size(), odm_scores.size(), odm_locs.size())
        batch_size = odm_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = odm_scores.size(2)

        # print(n_priors, predicted_locs.size(), predicted_scores.size())
        assert n_priors == odm_locs.size(1) == odm_scores.size(1)

        # Calculate ARM loss: offset smoothl1 + binary classification loss
        decoded_odm_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        # true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)
        ignored_priors = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # initial overlap
            ignored_overlap = find_jaccard_overlap(ignored_regions[i],
                                                   self.priors_xy)  # overlap of priors and ignored regions

            # For each prior, find if it overlaps with ignored regions overlap, return [value, indices]
            ignored_overlap_for_each_prior, _ = ignored_overlap.max(dim=0)

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
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            ignored_priors[i][ignored_overlap_for_each_prior >= self.theta] = 1

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            # true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]),
            #                                       xy_to_cxcy(decoded_arm_locs[i]))
            # true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
            true_locs[i] = boxes[i][object_for_each_prior]
            # print(odm_locs.size(), decoded_arm_locs.size())
            decoded_odm_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(odm_locs[i], self.priors_cxcy))

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0
        ignored_priors = ignored_priors > 0  # convert to uint8 for indexing

        # LOCALIZATION LOSS
        loc_loss = self.odm_loss(decoded_odm_locs[positive_priors].view(-1, 4),
                                 true_locs[positive_priors].view(-1, 4))
        # loc_loss = self.odm_loss(odm_locs[positive_priors].view(-1, 4),
        #                          true_locs_encoded[positive_priors].view(-1, 4))

        # CONFIDENCE LOSS
        # Number of positive and hard-negative priors per image
        # print('Classes:', self.n_classes, predicted_scores.size(), true_classes.size())
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.odm_cross_entropy(odm_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        # conf_loss_all = self.CELoss(odm_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, -1)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg[ignored_priors] = 0.  # negative priors in ignored regions are ignored
        conf_loss_neg, _ = conf_loss_neg.sort(dim=-1,
                                              descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
            self.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss

    def compute_segmentation_loss(self, attention_map, boxes):
        gt_map = torch.zeros(attention_map.size()).to(self.device)
        batch_size = attention_map.size(0)
        dims = torch.FloatTensor([self.fmap_dims['DB3'][1], self.fmap_dims['DB3'][0],
                                  self.fmap_dims['DB3'][1], self.fmap_dims['DB3'][0]]).unsqueeze(0)

        for i in range(batch_size):
            bboxes = torch.floor(boxes[i] * dims).int()
            for n in range(bboxes.size(0)):
                gt_map[i, :, bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]]

        loss = self.seg_loss(attention_map, gt_map)

        return loss

    def forward(self, odm_locs, odm_scores, attention_map, boxes, labels, ignored_regions):
        """
        :param attention_map:
        :param odm_locs: offset refinement prediction and multi-class classification scores from ODM
        :param odm_scores:
        :param boxes: gt bbox and labels
        :param labels:
        :return:
        """
        seg_loss = self.compute_segmentation_loss(attention_map, boxes)
        odm_loss = self.compute_odm_loss(odm_locs, odm_scores, boxes, labels, ignored_regions)

        # TOTAL LOSS
        return odm_loss + seg_loss
