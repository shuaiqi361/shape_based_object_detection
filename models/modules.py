import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding, default stride 1, shape unchanged
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def mish(inputs):
    """
        Applies the mish function element-wise:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))


class AdaptivePooling(nn.Module):
    def __init__(self, inplanes, outplanes=256, kernel_size=3, adaptive_size=64):
        super(AdaptivePooling, self).__init__()
        self.inplane = inplanes
        self.outplane = outplanes
        self.kernel_size = kernel_size
        self.adaptive_size = adaptive_size

        # multiple branches for feature extraction
        self.conv_astrous1 = nn.Conv2d(self.inplane, 256, kernel_size=self.kernel_size, padding=1, dilation=1)
        self.conv_astrous2 = nn.Conv2d(self.inplane, 256, kernel_size=self.kernel_size, padding=1, dilation=3)
        self.conv_astrous3 = nn.Conv2d(self.inplane, 256, kernel_size=self.kernel_size, padding=1, dilation=5)

        self.pool = nn.AdaptiveMaxPool2d(output_size=self.adaptive_size)
        self.transition = nn.Conv2d(256 * 3, self.outplane, kernel_size=1)
        self.act = mish

    def forward(self, x):
        astrous1 = self.conv_astrous1(x)
        astrous2 = self.conv_astrous2(x)
        astrous3 = self.conv_astrous3(x)
        feat = torch.cat([astrous1, astrous2, astrous3], dim=1)
        canonical_feat = self.pool(self.act(feat))

        feat = self.transition(canonical_feat)

        return self.act(feat)


class AttentionHead(nn.Module):
    def __init__(self, inplanes, reg_out, cls_out):
        super(AttentionHead, self).__init__()
        self.reg_out = reg_out
        self.cls_out = cls_out
        self.inplanes = inplanes
        self.reg_conv = nn.Conv2d(inplanes, inplanes, stride=1, kernel_size=3, padding=1)
        self.reg_conv_out = nn.Conv2d(inplanes, reg_out, stride=1, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(inplanes, inplanes, stride=1, kernel_size=3, padding=1)
        self.cls_conv_out = nn.Conv2d(inplanes, cls_out, stride=1, kernel_size=3, padding=1)
        self.n_classes = cls_out

    def forward(self, x):
        n_channels = x.size(1)
        assert n_channels == self.inplanes

        reg_mask = torch.sigmoid(self.reg_conv(x))
        cls_mask = torch.sigmoid(self.cls_conv(x))

        reg_feat = self.reg_conv_out(reg_mask * x)
        cls_feat = self.cls_conv_out(cls_mask * x)

        return reg_feat.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4), \
               cls_feat.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes)


class AttentionHeadSplit(nn.Module):
    def __init__(self, inplanes, reg_out, cls_out):
        super(AttentionHead, self).__init__()
        self.reg_out = reg_out
        self.cls_out = cls_out
        self.inplanes = inplanes
        self.n_channels = self.inplanes // 2
        self.reg_conv = nn.Conv2d(self.n_channels, self.n_channels, stride=1, kernel_size=3, padding=1)
        self.reg_conv_out = nn.Conv2d(self.n_channels, reg_out, stride=1, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(self.n_channels, self.n_channels, stride=1, kernel_size=3, padding=1)
        self.cls_conv_out = nn.Conv2d(self.n_channels, cls_out, stride=1, kernel_size=3, padding=1)
        self.n_classes = cls_out

    def forward(self, x):
        assert self.inplanes == x.size(1)
        reg_feat = x[:, :self.n_channels, :, :]
        cls_feat = x[:, self.n_channels:, :, :]

        reg_mask = torch.sigmoid(self.reg_conv(x))
        cls_mask = torch.sigmoid(self.cls_conv(x))

        reg_feat = self.reg_conv_out(reg_mask * reg_feat)
        cls_feat = self.cls_conv_out(cls_mask * cls_feat)

        return reg_feat.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4), \
               cls_feat.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes)



