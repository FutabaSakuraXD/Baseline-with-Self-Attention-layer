from __future__ import absolute_import

import math
import torch
from torch.nn import Parameter
from torch import cat
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

__all__ = ['ResNet', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'EnhancedBaseline', 'ide']


def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = False

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        self.base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.base.fc.in_features
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  # default dropout rate 0.5

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.base.fc = add_block

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            # print(name, module)
            if name == 'avgpool':
                break
            # print(x)
            x = module(x)

        pool5 = F.avg_pool2d(x, x.size()[2:])
        pool5 = pool5.view(pool5.size(0), -1)

        return pool5

##################################################################################
# Self-Attention Layer
##################################################################################

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out



class EnhancedBaseline(nn.Module):
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def __init__(self, num_classes, num_features):
        super(ft_net, self).__init__()
        self.memorybank = torch.zeros(2, 1 * num_classes, 2048).cuda()
        self.memorybank = Parameter(self.memorybank)

        self.Weight_of_Global_Feature_Weight = 1
        self.Weight_of_Local_Feature_Weight = 1

        for p in self.parameters():
            p.requires_grad = False

        # Enhancing block, containing one channel attention and one pixel attention and other layers
        self.Self_Attn_AfterRes50Layer1 = Self_Attn(in_dim=256, activation='relu')
        self.Self_Attn_AfterRes50Layer3 = Self_Attn(in_dim=1024, activation='relu')

        # MGN structure
        resnet = torchvision.models.resnet50(pretrained=True)
        self.model = resnet
        self.backbone1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.backbone2 = nn.Sequential(
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        pool2d = nn.MaxPool2d

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, num_features, 1, bias=False), nn.BatchNorm2d(num_features), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(num_features, num_classes)
        self.fc_id_2048_1 = nn.Linear(num_features, num_classes)
        self.fc_id_2048_2 = nn.Linear(num_features, num_classes)

        self.fc_id_256_1_0 = nn.Linear(num_features, num_classes)
        self.fc_id_256_1_1 = nn.Linear(num_features, num_classes)
        self.fc_id_256_2_0 = nn.Linear(num_features, num_classes)
        self.fc_id_256_2_1 = nn.Linear(num_features, num_classes)
        self.fc_id_256_2_2 = nn.Linear(num_features, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    def forward(self, x):
        # x = self.backbone1(x)

        x = self.backbone1(x)
        x = self.Self_Attn_AfterRes50Layer1(x) # channel 256
        x = self.backbone2(x)
        x = self.Self_Attn_AfterRes50Layer3(x) # channel 1024

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        # print("The shape of p1, p2, p3 :\n" + str(p1.shape) + "\t" + str(p2.shape) + "\t" + str(p3.shape))

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)
        # print("The shape of zg_p1, zg_p2, zg_p3 :\n" + str(zg_p1.shape) + "\t" + str(zg_p2.shape) + "\t" + str(zg_p3.shape))

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]
        # print("The shape of zp2, z0_p2, z1_p2 :\n" + str(zp2.shape) + "\t" + str(z0_p2.shape) + "\t" + str(z1_p2.shape))

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        # print("The shape of zp3, z0_p3, z1_p3, z2_p3 :\n" + str(zp3.shape) + "\t" + str(z0_p3.shape) + "\t" + str(z1_p3.shape) + "\t" + str(z2_p3.shape))

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)
        # print("Feature vector shape after reduction_i:\t" + str(fg_p1.shape))
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        # print("Since fc_id_2048 and fc_id_256 would output the same size, here we only show one line of'me\n" + str(l_p1.shape))

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        fg_p1 = self.Weight_of_Global_Feature_Weight * fg_p1
        fg_p2 = self.Weight_of_Global_Feature_Weight * fg_p2
        fg_p3 = self.Weight_of_Global_Feature_Weight * fg_p3
        f0_p2 = self.Weight_of_Local_Feature_Weight * f0_p2
        f1_p2 = self.Weight_of_Local_Feature_Weight * f1_p2
        f0_p3 = self.Weight_of_Local_Feature_Weight * f0_p3
        f1_p3 = self.Weight_of_Local_Feature_Weight * f1_p3
        f2_p3 = self.Weight_of_Local_Feature_Weight * f2_p3
        return fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3 \
            , l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


def ide(**kwargs):
    return ft_net(**kwargs)
