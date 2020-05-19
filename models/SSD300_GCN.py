from torch import nn
import torch
import torch.nn.functional as F
import models.functional as Func
from math import sqrt
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SmoothL1Loss, SigmoidFocalLoss, LabelSmoothingLoss
from metrics import find_jaccard_overlap


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
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

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
        out = Func.mish(self.conv1_1(image))  # (N, 64, 300, 300)
        out = Func.mish(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = Func.mish(self.conv2_1(out))  # (N, 128, 150, 150)
        out = Func.mish(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = Func.mish(self.conv3_1(out))  # (N, 256, 75, 75)
        out = Func.mish(self.conv3_2(out))  # (N, 256, 75, 75)
        out = Func.mish(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = Func.mish(self.conv4_1(out))  # (N, 512, 38, 38)
        out = Func.mish(self.conv4_2(out))  # (N, 512, 38, 38)
        out = Func.mish(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = Func.mish(self.conv5_1(out))  # (N, 512, 19, 19)
        out = Func.mish(self.conv5_2(out))  # (N, 512, 19, 19)
        out = Func.mish(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = Func.mish(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = Func.mish(self.conv7(out))  # (N, 1024, 19, 19)

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
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
                # nn.init.normal_(c.weight)
                if c.bias is not None:
                    nn.init.normal_(c.bias)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = Func.mish(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = Func.mish(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = Func.mish(self.conv9_1(out))  # (N, 128, 10, 10)
        out = Func.mish(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = Func.mish(self.conv10_1(out))  # (N, 128, 5, 5)
        out = Func.mish(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = Func.mish(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = Func.mish(self.conv11_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        # print(conv8_2_feats.size(), conv9_2_feats.size(), conv10_2_feats.size(), conv11_2_feats.size())
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the function.
        """
        return Func.mish(x)


class GCNHead(nn.Module):
    def __init__(self, inplanes, n_classes, n_boxes, n_points, node_dim, config):
        super(GCNHead, self).__init__()
        self.inplanes = inplanes
        self.n_classes = n_classes
        self.n_points = n_points
        self.node_dim = node_dim
        self.n_boxes = n_boxes
        self.config = config
        self.device = config.device
        self.fmap_dims = {'conv4_3': 38,
                          'conv7': 19,
                          'conv8_2': 10,
                          'conv9_2': 5,
                          'conv10_2': 3,
                          'conv11_2': 1}

        self.encoder = nn.Conv2d(self.inplanes, self.n_boxes * self.n_points * self.node_dim, kernel_size=3, padding=1)
        self.decoder_cls = nn.Linear(self.n_points * self.node_dim, self.n_classes)
        self.decoder_reg = nn.Linear(self.n_points * self.node_dim, self.n_points * 2)
        self.adjacency_mat, self.degree_matrix = self.create_graph(self.n_points)  # Both are 9x9 matrix

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d) or isinstance(c, torch.nn.Linear):
                nn.init.xavier_normal_(c.weight)
                # nn.init.normal_(c.weight)
                if c.bias is not None:
                    nn.init.normal_(c.bias)

    def create_graph(self, n_nodes):
        # [1.0, 0.5, 0.25, 0.125, 0.0625, 0.125, 0.25, 0.5] for 9 nodes
        edge_len = [0.5 ** min(i, n_nodes - i) for i in range(n_nodes)]
        # print(edge_len)
        # print(n_nodes)
        adj_mat_half = torch.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            count = 0
            for j in range(i, n_nodes):
                adj_mat_half[i, j] = edge_len[count]
                count += 1

        adj_mat = adj_mat_half + torch.transpose(adj_mat_half, 0, 1) - torch.eye(n_nodes)
        deg_mat = torch.diag(adj_mat.sum(dim=1) ** -1)
        # print(adj_mat.data)
        # print(deg_mat.data)
        # exit()

        return adj_mat.to(self.device), deg_mat.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        encoded_feats = Func.mish(self.encoder(x)).permute(0, 2, 3, 1).contiguous()  # bs, feat, feat, 4 * 9 * 16
        graph_feat = torch.matmul(torch.matmul(encoded_feats.view(batch_size, -1, self.node_dim, self.n_points),
                                               self.adjacency_mat.unsqueeze(0)), self.degree_matrix.unsqueeze(0))
        # print(graph_feat.size())
        # print(batch_size)
        cls = self.decoder_cls(graph_feat.view(batch_size, -1, self.n_points * self.node_dim))

        reg = self.decoder_reg(graph_feat.view(batch_size, -1, self.n_points * self.node_dim))

        return reg, cls


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 22536 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 22536 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, config, n_points=9, node_dim=16):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes
        self.n_points = n_points
        self.node_dim = node_dim
        self.config = config
        self.device = config.device

        # Number of prior-boxes we are considering per position in each feature map
        self.sigmoid = nn.Sigmoid()
        self.n_boxes = 4
        self.scale_param = nn.Parameter(torch.FloatTensor([0.02, 0.04, 0.08, 0.16, 0.32, 0.64]))
        self.aspect_param = nn.Parameter(torch.FloatTensor([[1, 1], [2, 2], [4, 4], [8, 8], [16, 16], [32, 32]]))

        # regularize to the same dim, so that the GCN head and regression head can be shared
        self.node_conv4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.node_conv7 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.node_conv8_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.node_conv9_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.node_conv10_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.node_conv11_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.loc_conv = nn.Conv2d(256, self.n_boxes * 4, kernel_size=3, padding=1)
        # self.loc_conv7 = nn.Conv2d(256, self.n_boxes * 4, kernel_size=3, padding=1)
        # self.loc_conv8 = nn.Conv2d(256, self.n_boxes * 4, kernel_size=3, padding=1)
        # self.loc_conv9 = nn.Conv2d(256, self.n_boxes * 4, kernel_size=3, padding=1)
        # self.loc_conv10 = nn.Conv2d(256, self.n_boxes * 4, kernel_size=3, padding=1)
        # self.loc_conv11 = nn.Conv2d(256, self.n_boxes * 4, kernel_size=3, padding=1)

        self.cls_conv = GCNHead(256, self.n_classes, self.n_boxes, self.n_points, self.node_dim, self.config)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
                # nn.init.normal_(c.weight)
                if c.bias is not None:
                    nn.init.normal_(c.bias)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats,
                conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.

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

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        conv4_3_feats = Func.mish(self.node_conv4_3(conv4_3_feats))
        conv7_feats = Func.mish(self.node_conv7(conv7_feats))
        conv8_2_feats = Func.mish(self.node_conv8_2(conv8_2_feats))
        conv9_2_feats = Func.mish(self.node_conv9_2(conv9_2_feats))
        conv10_2_feats = Func.mish(self.node_conv10_2(conv10_2_feats))
        conv11_2_feats = Func.mish(self.node_conv11_2(conv11_2_feats))

        l_conv4_3 = self.loc_conv(conv4_3_feats).permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4) * self.scale_param[0].view(1, 1, 1)

        l_conv7 = self.loc_conv(conv7_feats).permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4) * self.scale_param[1].view(1, 1, 1)

        l_conv8_2 = self.loc_conv(conv8_2_feats).permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4) * self.scale_param[2].view(1, 1, 1)

        l_conv9_2 = self.loc_conv(conv9_2_feats).permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4) * self.scale_param[3].view(1, 1, 1)

        l_conv10_2 = self.loc_conv(conv10_2_feats).permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4) * self.scale_param[4].view(1, 1, 1)

        l_conv11_2 = self.loc_conv(conv11_2_feats).permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4) * self.scale_param[5].view(1, 1, 1)

        # Predict classes in localization boxes and features for each node
        # reg: batch, n_priors, n_points, 2
        # cls: batch, n_priors, n_classes
        reg_conv4_3, cls_conv4_3 = self.cls_conv(conv4_3_feats)
        reg_conv4_3 = self.sigmoid(reg_conv4_3 *
                                   (self.aspect_param[0].view(1, 1, 2).repeat(1, 1, self.n_points))) * 2.0 - 1

        reg_conv7, cls_conv7 = self.cls_conv(conv7_feats)
        reg_conv7 = self.sigmoid(reg_conv7 *
                                 (self.aspect_param[1].view(1, 1, 2).repeat(1, 1, self.n_points))) * 2.0 - 1

        reg_conv8_2, cls_conv8_2 = self.cls_conv(conv8_2_feats)
        reg_conv8_2 = self.sigmoid(reg_conv8_2 *
                                   (self.aspect_param[2].view(1, 1, 2).repeat(1, 1, self.n_points))) * 2.0 - 1

        reg_conv9_2, cls_conv9_2 = self.cls_conv(conv9_2_feats)
        reg_conv9_2 = self.sigmoid(reg_conv9_2 *
                                   (self.aspect_param[3].view(1, 1, 2).repeat(1, 1, self.n_points))) * 2.0 - 1

        reg_conv10_2, cls_conv10_2 = self.cls_conv(conv10_2_feats)
        reg_conv10_2 = self.sigmoid(reg_conv10_2 *
                                    (self.aspect_param[4].view(1, 1, 2).repeat(1, 1, self.n_points))) * 2.0 - 1

        reg_conv11_2, cls_conv11_2 = self.cls_conv(conv11_2_feats)
        reg_conv11_2 = self.sigmoid(reg_conv11_2 *
                                    (self.aspect_param[5].view(1, 1, 2).repeat(1, 1, self.n_points))) * 2.0 - 1

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2],
                         dim=1)
        # print(reg_conv4_3.size(), reg_conv7.size(), reg_conv8_2.size(), reg_conv9_2.size(), reg_conv10_2.size(), reg_conv11_2.size())
        regs_offset = torch.cat([reg_conv4_3, reg_conv7, reg_conv8_2, reg_conv9_2, reg_conv10_2, reg_conv11_2], dim=1)
        classes_scores = torch.cat([cls_conv4_3, cls_conv7, cls_conv8_2, cls_conv9_2, cls_conv10_2, cls_conv11_2],
                                   dim=1)

        return locs, regs_offset, classes_scores


class SSD300GCN(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes, config):
        super(SSD300GCN, self).__init__()
        self.device = config.device
        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes, config, n_points=9, node_dim=16)
        self.n_points = self.pred_convs.n_points
        self.node_dim = self.pred_convs.node_dim
        self.shape_weight = nn.Parameter(data=torch.ones(1, 1, 4) * 0.5, requires_grad=True)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors_conv4_3 = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors_conv4_3, 20.)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

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
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 8, 8),  (N, 256, 4, 4), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, regs_offset, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats,
                                                            conv9_2_feats, conv10_2_feats,
                                                            conv11_2_feats)
        predicted_points = self.offset2points(regs_offset)
        bbox_from_shape = self.point2bbox(predicted_points)
        bbox_from_reg = torch.zeros(bbox_from_shape.size()).to(self.device)
        for i in range(bbox_from_reg.size(0)):
            bbox_from_reg[i] = cxcy_to_xy(gcxgcy_to_cxcy(locs[i], self.priors_cxcy))

        # leverage the prediction on both predicted offsets and shapes
        final_bbox = bbox_from_shape * self.shape_weight.detach() + bbox_from_reg * (1 - self.shape_weight)
        # final_bbox = bbox_from_reg

        return final_bbox, classes_scores, predicted_points

    def point2bbox(self, predicted_points):
        """
        predicted_points: batch_size, n_priors, 9, 2
        """
        predicted_tl = predicted_points.min(dim=2)[0]  # batch_size, n_priors, 2  (x1, y1)
        predicted_br = predicted_points.max(dim=2)[0]  # batch_size, n_priors, 2  (x2, y2)
        predicted_bbox = torch.cat([predicted_tl, predicted_br], dim=2)  # batch_size, n_priors, 4

        return predicted_bbox

    def offset2points(self, predicted_offsets):
        """
        predicted_offsets: batch_size, n_priors, 9 * 2
        """
        batch_size = predicted_offsets.size(0)
        predicted_points = predicted_offsets.view(batch_size, -1, self.n_points, 2) + \
                           self.priors_cxcy[:, :2].view(1, -1, 1, 2)

        return predicted_points

    def create_prior_boxes(self):
        """
        Create the 22536 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (22536, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 0.5],
                         'conv8_2': [1., 2., 0.5],
                         'conv9_2': [1., 2., 0.5],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                # use the geometric mean to calculate the additional scale for each level of fmap
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last object scale, there is no "next" scale
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device)
        prior_boxes.clamp_(0, 1)

        return prior_boxes


class MultiBoxGCNLoss300(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, threshold=0.5, neg_pos_ratio=3):
        super(MultiBoxGCNLoss300, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.config = config

        self.smooth_l1 = SmoothL1Loss
        self.Diou_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Diou')
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.Focal_loss = SigmoidFocalLoss(gamma=2., alpha=.25, config=config)
        self.SoftCE = LabelSmoothingLoss(classes=self.n_classes, smoothing=config.smooth_threshold, reduce=False)

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

        # decoded_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 22536, 4)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 22536, 4)
        # true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 22536, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 22536)
        true_neg_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 22536)

        # For each image
        for i in range(batch_size):
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 22536)

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (22536)
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
            # label_for_each_prior[overlap_for_each_prior < self.threshold] = -1  # label in 0.4-0.5 is not used
            # label_for_each_prior[overlap_for_each_prior < self.threshold - 0.1] = 0
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            label_for_each_prior_neg_used[overlap_for_each_prior < self.threshold - 0.1] = -1

            # Store
            true_classes[i] = label_for_each_prior
            true_neg_classes[i] = label_for_each_prior_neg_used

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = boxes[i][object_for_each_prior]
            # true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
            # decoded_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0
        negative_priors = true_neg_classes == -1

        # LOCALIZATION LOSS
        loc_loss = self.Diou_loss(predicted_locs[positive_priors].view(-1, 4),
                                  predicted_locs[positive_priors].view(-1, 4))
        # if self.config.reg_loss.upper() == 'IOU':
        #     loc_loss = self.Diou_loss(decoded_locs[positive_priors].view(-1, 4),
        #                               true_locs[positive_priors].view(-1, 4))
        # else:
        #     loc_loss = self.smooth_l1(predicted_locs[positive_priors].view(-1, 4),
        #                               true_locs_encoded[positive_priors].view(-1, 4))

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
            # To do this, sort ONLY negative priors in each image in order of decreasing loss and
            # take top n_hard_negatives
            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            # conf_loss_neg[~negative_priors] = 0.
            conf_loss_neg[positive_priors] = 0.
            conf_loss_neg, _ = conf_loss_neg.sort(dim=-1, descending=True)  # (N, 8732), sorted by decreasing hardness
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
                self.device)  # (N, 8732)
            hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

            conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
