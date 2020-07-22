from torch import nn
from operators.Deformable_convolution import DeformConv2d
import torch.nn.functional as F
from math import sqrt
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, FocalLoss, focal_loss, SmoothL1Loss, SigmoidFocalLoss
from metrics import find_jaccard_overlap
from operators.iou_utils import find_distance
import math


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
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

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
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300) 360, 640
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150) 180, 320
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75) 90, 160
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38) 45, 80
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        # print(conv4_3_feats.size(), conv7_feats.size())
        return conv4_3_feats, conv7_feats

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

        print("Loading pre-trained VGG16 base model.")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_1_gn = nn.GroupNorm(16, 256)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        self.conv8_2_gn = nn.GroupNorm(16, 512)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_1_gn = nn.GroupNorm(16, 128)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        self.conv9_2_gn = nn.GroupNorm(16, 256)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_1_gn = nn.GroupNorm(16, 128)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv10_2_gn = nn.GroupNorm(16, 256)

        # self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        # self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        #
        # self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
        # self.conv12_2 = nn.Conv2d(128, 256, kernel_size=2)

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

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1_gn(self.conv8_1(conv7_feats)))  # (N, 256, 16, 16)
        out = F.relu(self.conv8_2_gn(self.conv8_2(out)))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1_gn(self.conv9_1(out)))  # (N, 128, 8, 8)
        out = F.relu(self.conv9_2_gn(self.conv9_2(out)))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1_gn(self.conv10_1(out)))  # (N, 128, 4, 4)
        out = F.relu(self.conv10_2_gn(self.conv10_2(out)))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        # out = F.relu(self.conv11_1(out))  # (N, 128, 2, 2)
        # out = F.relu(self.conv11_2(out))
        # conv11_2_feats = out  # (N, 256, 1, 1)
        #
        # out = F.relu(self.conv12_1(out))
        # out = F.relu(self.conv12_2(out))
        # conv12_2_feats = out  # (N, 256, 1, 1)

        # Higher-level feature maps
        # print(conv8_2_feats.size(), conv9_2_feats.size(), conv10_2_feats.size(), conv11_2_feats.size())
        return conv8_2_feats, conv9_2_feats, conv10_2_feats  # , conv11_2_feats, conv12_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 22536 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 22536 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 1,
                   'conv7': 1,
                   'conv8_2': 1,
                   'conv9_2': 1,
                   'conv10_2': 1,
                   'conv11_2': 1,
                   'conv12_2': 1}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        # self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)
        # self.loc_conv12_2 = nn.Conv2d(256, n_boxes['conv12_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        # self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
        # self.cl_conv12_2 = nn.Conv2d(256, n_boxes['conv12_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                # nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.weight, val=0.)
                if c.bias is not None:
                    nn.init.constant_(c.bias, val=0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats,
                conv10_2_feats):
        """
        Forward propagation.

        :param conv12_2_feats:
        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 64, 64)
        :param conv5_3_feats: conv5_3 feature map, a tensor of dimensions (N, 512, 32, 32)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 16, 16)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 8, 8)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 4, 4)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 22536 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 64, 64)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 64, 64, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 16384, 4), there are a total 16384 boxes

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 16, 16)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 16, 16, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 1536, 4), there are a total 1536 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 8, 8)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 8, 8, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 384, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 4, 4)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 4, 4, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 96, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        # l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        # l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        # l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)
        #
        # l_conv12_2 = self.loc_conv12_2(conv12_2_feats)  # (N, 16, 1, 1)
        # l_conv12_2 = l_conv12_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        # l_conv12_2 = l_conv12_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)

        # c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        # c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        # c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)
        #
        # c_conv12_2 = self.cl_conv12_2(conv12_2_feats)
        # c_conv12_2 = c_conv12_2.permute(0, 2, 3, 1).contiguous()
        # c_conv12_2 = c_conv12_2.view(batch_size, -1, self.n_classes)

        # A total of 22536 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2],
                         dim=1).contiguous()
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2],
                                   dim=1).contiguous()

        return locs, classes_scores


class ATSSSSD512(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes, config, prior=0.01):
        super(ATSSSSD512, self).__init__()
        self.device = config.device
        self.n_classes = n_classes - 1
        self.base = VGGBase()
        self.prior = prior
        # self.disable_parameter_requires_grad(self.base)
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(self.n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors_conv4_3 = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors_conv4_3, 10.)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

        # initialization
        self.pred_convs.cl_conv4_3.weight.data.fill_(0)
        self.pred_convs.cl_conv4_3.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.pred_convs.cl_conv7.weight.data.fill_(0)
        self.pred_convs.cl_conv7.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.pred_convs.cl_conv8_2.weight.data.fill_(0)
        self.pred_convs.cl_conv8_2.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.pred_convs.cl_conv9_2.weight.data.fill_(0)
        self.pred_convs.cl_conv9_2.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))
        self.pred_convs.cl_conv10_2.weight.data.fill_(0)
        self.pred_convs.cl_conv10_2.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))

    def disable_parameter_requires_grad(self, m):
        for param in m.parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: 22536 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 64, 64), (N, 1024, 32, 32), (N, 1024, 16, 16)

        # # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 64, 64)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 64, 64)
        conv4_3_feats = conv4_3_feats * self.rescale_factors_conv4_3  # (N, 512, 64, 64)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 8, 8),  (N, 256, 4, 4), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats,
                                               conv9_2_feats, conv10_2_feats)  # (N, 22536, 4), (N, 22536, n_classes)

        # print(conv4_3_feats.size(), conv8_2_feats.size(), conv10_2_feats.size(), conv12_2_feats.size())
        return locs, classes_scores

    def offset2bbox(self, predicted_offsets):
        predicted_bbox = torch.cat([self.priors_cxcy[:, :2].unsqueeze(0).repeat(
            predicted_offsets.size(0), 1, 1) - predicted_offsets[:, :, :2],
                                    self.priors_cxcy[:, :2].unsqueeze(0).repeat(
                                        predicted_offsets.size(0), 1, 1) + predicted_offsets[:, :, 2:]], dim=2)
        return predicted_bbox

    def create_prior_boxes(self):
        """
        Create the 22536 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (22536, 4)
        """
        # fmap_dims = {'conv4_3': [38, 38],
        #              'conv7': [19, 19],
        #              'conv8_2': [10, 10],
        #              'conv9_2': [5, 5],
        #              'conv10_2': [3, 3]}
        # fmap_dims = {'conv4_3': [90, 90],
        #              'conv7': [45, 45],
        #              'conv8_2': [23, 23],
        #              'conv9_2': [12, 12],
        #              'conv10_2': [6, 6]}
        fmap_dims = {'conv4_3': [80, 80],
                     'conv7': [40, 40],
                     'conv8_2': [20, 20],
                     'conv9_2': [10, 10],
                     'conv10_2': [5, 5]}

        obj_scales = {'conv4_3': 0.02,
                      'conv7': 0.04,
                      'conv8_2': 0.08,
                      'conv9_2': 0.16,
                      'conv10_2': 0.32}

        aspect_ratios = {'conv4_3': [1.],
                         'conv7': [1.],
                         'conv8_2': [1.],
                         'conv9_2': [1.],
                         'conv10_2': [1.]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []
        scale_factor = [1.]
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


class ATSSSSD512Loss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, n_candidates=9):
        super(ATSSSSD512Loss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = [cxcy_to_xy(prior) for prior in self.priors_cxcy]
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes - 1
        self.n_candidates = n_candidates

        fmap_dims = {'conv4_3': [80, 80],
                     'conv7': [40, 40],
                     'conv8_2': [20, 20],
                     'conv9_2': [10, 10],
                     'conv10_2': [5, 5]}

        self.prior_split_points = [0]
        fmap_keys = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2']
        for k in fmap_keys:
            self.prior_split_points.append(self.prior_split_points[-1] + fmap_dims[k][0] * fmap_dims[k][1])

        # self.prior_split_points = [0, 1444, 1805, 1905, 1930, 1939]
        # self.prior_split_points = [0, 8160, 10200, 10710, 10845, 10885]
        # self.prior_split_points = [0, 10000, 12500, 13125, 13294, 13343]
        # self.regression_loss = SmoothL1Loss(reduction='mean')
        self.regression_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Ciou')
        # self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.FocalLoss = SigmoidFocalLoss(gamma=2.0, alpha=0.25, config=config)
        # self.FocalLoss = focal_loss

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        :param predicted_scores: list of predicted class scores for each feature level
        :param predicted_locs: list of predicted bboxes for each feature level
        :param boxes: gt
        :param labels: gt
        :return
        """
        # print('locs:', predicted_locs[0, 0:2, :], 'scores:', predicted_scores[0, 0:2, :])
        n_levels = len(self.priors_cxcy)
        batch_size = predicted_locs.size(0)
        n_priors = np.sum([prior.size(0) for prior in self.priors_cxcy])
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
        # overall training samples for all images in a batch
        decoded_locs = list()  # length is the batch size
        true_locs = list()
        true_classes = list()
        predicted_class_scores = list()

        # For each image
        for i in range(batch_size):
            image_bboxes = boxes[i]
            # print('image_boxes size: ', image_bboxes.size())
            batch_split_predicted_locs = []
            batch_split_predicted_scores = []
            # split the predictions according to the feature pyramid dimension
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
            # candidates for positive samples, use to calculate the IOU threshold
            positive_samples_idx = list()
            positive_overlaps = list()
            overlap = list()  # for all
            for level in range(n_levels):
                distance = find_distance(xy_to_cxcy(image_bboxes), self.priors_cxcy[level])  # n_bboxes, n_priors
                # for each object bbox, find the top_k closest prior boxes
                _, top_idx_level = torch.topk(-1. * distance, min(self.n_candidates, distance.size(1)), dim=1)
                # print(distance.size(), top_idx_level.size())
                # print('Topk distance:', distance[:1, :5])
                # print(top_idx_level)
                # exit()

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
                # print(positive_overlaps[0])
                # exit()
                overlap.append(overlap_level)

            positive_overlaps_cat = torch.cat(positive_overlaps, dim=1)  # n_bboxes, n_priors * 6 levels
            # print('positive_overlaps_cat shape: ', positive_overlaps_cat.size())
            overlap_mean = torch.mean(positive_overlaps_cat, dim=1)
            overlap_std = torch.std(positive_overlaps_cat, dim=1)
            # print(overlap_mean, overlap_std)
            iou_threshold = overlap_mean + overlap_std  # n_bboxes, for each object, we have one threshold
            # print([p for p in positive_overlaps])
            # print('threshold: ', iou_threshold)
            # exit()

            # one prior can only be associated to one gt object
            # For each prior, find the object that has the maximum overlap, return [value, indices]
            # overlap = torch.cat(overlap, dim=1)
            true_classes_level = list()
            true_locs_level = list()
            positive_priors = list()  # For all levels
            decoded_locs_level = list()
            for level in range(n_levels):
                positive_priors_per_level = torch.zeros((image_bboxes.size(0), self.priors_cxcy[level].size(0)),
                                                        dtype=torch.uint8).to(self.device)  # indexing, (n,)
                # print(positive_priors_per_level.size())
                # print(level, 'total decoded priors shape: ', total_decoded_locs.size())
                # overlap_for_each_prior, object_for_each_prior = overlap[level].max(dim=0)
                # overlap_for_each_object, prior_for_each_object = overlap[level].max(dim=1)
                for ob in range(image_bboxes.size(0)):
                    for c in range(len(positive_samples_idx[level][ob])):
                        # print(ob, c, 'Range for c: ', len(positive_samples_idx[level][ob]))
                        current_iou = positive_overlaps[level][ob, c]
                        current_bbox = image_bboxes[ob, :]
                        current_prior = self.priors_cxcy[level][positive_samples_idx[level][ob, c], :]
                        # print('iou thres, bbox prior', current_iou, iou_threshold[ob], current_bbox, current_prior)
                        # exit()

                        if current_iou > iou_threshold[ob]:
                            if current_bbox[0] < current_prior[0] < current_bbox[2] \
                                    and current_bbox[1] < current_prior[1] < current_bbox[3]:
                                # print('------------------------------------------------------------------')
                                positive_priors_per_level[ob, positive_samples_idx[level][ob, c]] = 1
                                # if current_iou == overlap_for_each_prior[positive_samples_idx[level][ob, c]]:
                                # print(label_for_each_prior_per_level[positive_samples_idx[level][ob, c]])
                                # label_for_each_prior_per_level[positive_samples_idx[level][ob, c]] = labels[i][ob]
                                # print(label_for_each_prior_per_level[positive_samples_idx[level][ob, c]])
                                # print('Details:')
                                # print(positive_samples_idx[level][ob, c], labels[i][ob])
                                # exit()
                positive_priors.append(positive_priors_per_level)

            for level in range(n_levels):  # this is the loop for find the best object for each prior,
                # because one prior could match with more than one objects
                label_for_each_prior_per_level = torch.zeros((self.priors_cxcy[level].size(0)),
                                                             dtype=torch.long).to(self.device)
                true_locs_per_level = list()  # only for positive candidates in the predictions
                decoded_locs_per_level = list()
                total_decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(batch_split_predicted_locs[level], self.priors_cxcy[level]))

                for c in range(positive_samples_idx[level].size(1)):
                # for c in range(self.priors_cxcy[level].size(0)):  # loop over each prior in each level
                    current_max_iou = 0.
                    current_max_iou_ob = -1  # index for rows: (n_ob, n_prior)
                    for ob in range(image_bboxes.size(0)):
                        # print(positive_samples_idx[level].size(1), 'positive_priors_per_level shape: ', positive_priors_per_level.size())
                        # print('positive_priors size per level:', positive_priors[level].size())
                        # print(ob)
                        # print(positive_samples_idx[level][ob, c])
                        if positive_priors[level][ob, positive_samples_idx[level][ob, c]] == 1:
                        # if positive_priors[level][ob, c] == 1:
                            if overlap[level][ob, positive_samples_idx[level][ob, c]] > current_max_iou:
                                current_max_iou_ob = ob
                                current_max_iou = overlap[level][ob, positive_samples_idx[level][ob, c]]

                    # if current_max_iou_ob > -1 and current_max_iou > 0. and label_for_each_prior_per_level[positive_samples_idx[level][current_max_iou_ob, c]] == 0:
                    if current_max_iou_ob > -1 and current_max_iou > 0.:
                        # print('Assign positive:', current_max_iou, current_max_iou_ob)
                        # print(overlap[level][:, positive_samples_idx[level][ob, c]])
                        # exit()
                        temp_true_locs = image_bboxes[current_max_iou_ob, :].unsqueeze(0)  # (1, 4)
                        # temp_decoded_locs = total_decoded_locs[c, :].unsqueeze(0)
                        temp_decoded_locs = total_decoded_locs[positive_samples_idx[level][current_max_iou_ob, c], :].unsqueeze(0)  # (1, 4)
                        label_for_each_prior_per_level[c] = labels[i][current_max_iou_ob]
                        true_locs_per_level.append(temp_true_locs)
                        decoded_locs_per_level.append(temp_decoded_locs)
                        # print(level, 'Assignment for labels: ', labels[i][current_max_iou_ob], ' at ', positive_samples_idx[level][current_max_iou_ob, c], ' with length ', len(true_locs_per_level))

                # if len(true_locs_per_level) > 0:
                #     print('show: ', (label_for_each_prior_per_level > 0).sum().float(), len(true_locs_per_level), torch.cat(true_locs_per_level, dim=0).size())
                #     assert (label_for_each_prior_per_level > 0).sum().float() == len(true_locs_per_level)
                # print(label_for_each_prior_per_level.size(), len(self.priors_cxcy[level]))
                # assert label_for_each_prior_per_level.size(0) == self.priors_cxcy[level].size(0)
                if len(true_locs_per_level) > 0:
                    true_locs_level.append(torch.cat(true_locs_per_level, dim=0).view(-1, 4))  # (1, n_l * 4)
                    decoded_locs_level.append(torch.cat(decoded_locs_per_level, dim=0).view(-1, 4))
                    # assert torch.cat(decoded_locs_per_level, dim=0).view(-1, 4).size(0) == torch.cat(
                    #     true_locs_per_level,
                    #     dim=0).view(-1, 4).size(0)
                true_classes_level.append(label_for_each_prior_per_level)
                # assert len(label_for_each_prior_per_level) == len(batch_split_predicted_locs[level])

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
        # print('true locs:', true_locs[:10, :])
        # # # print('true classes:', true_classes[0:1000])
        # # print('Number of positive priors:', (true_classes > 0).sum().float())
        # print('decoded locs:', decoded_locs[:10, :])
        # #
        # exit()

        # LOCALIZATION LOSS
        loc_loss = self.regression_loss(decoded_locs, true_locs)

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum().float()

        # First, find the loss for all priors
        # conf_loss = self.FocalLoss(predicted_scores, true_classes, device=self.device) / true_classes.size(0) * 1.
        conf_loss = self.FocalLoss(predicted_scores, true_classes) / n_positives
        # conf_loss = self.FocalLoss(predicted_scores, true_classes) / len(true_classes)

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
