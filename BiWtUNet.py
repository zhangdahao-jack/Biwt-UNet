import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from einops import rearrange, repeat  # vit
from einops.layers.torch import Rearrange  # vit
from torch.nn import init
import pywt
from torch.autograd import Variable
import math

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class wt(nn.Module):
    def __init__(self, wave='db1'):
        super(wt, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        filters = torch.stack([
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,  # LL
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),        # LH
        ], dim=0)  # shape: [2, k, k]

        filters = filters.unsqueeze(1)  # [2, 1, k, k]
        self.register_buffer('filters', filters)

    def forward(self, x):
        batch, channels, h, w = x.shape
        device = x.device
        out = torch.zeros(batch, 2 * channels, h // 2, w // 2, device=device)

        for i in range(channels):
            out[:, 2 * i:2 * i + 2] = F.conv2d(
                x[:, i:i + 1], self.filters, stride=2
            )
        return out
class iwt(nn.Module):
    def __init__(self, wave='db1'):
        super(iwt, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        inv_filters = torch.stack([
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,  # LL
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),        # LH
        ], dim=0)  # shape: [2, k, k]

        inv_filters = inv_filters.unsqueeze(1)  # [2, 1, k, k]
        self.register_buffer('inv_filters', inv_filters)

    def forward(self, x):
        batch, channels, h, w = x.shape
        device = x.device
        out = torch.zeros(batch, channels // 2, h * 2, w * 2, device=device)

        for i in range(channels // 2):
            out[:, i:i + 1] = F.conv_transpose2d(
                x[:, 2 * i:2 * i + 2], self.inv_filters, stride=2
            )
        return out


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x


class tongdao(nn.Module):  #处理通道部分   函数名就是拼音名称
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准

class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励

class hebing(nn.Module):    #函数名为合并, 意思是把空间和通道分别提取的特征合并起来
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励


class MDFA(nn.Module):                       ##多尺度空洞融合注意力模块。
    def __init__(self, dim_in, dim_out, rate=3, bn_mom=0.1):# 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
        super(MDFA, self).__init__()
        self.branch1 = nn.Sequential(# 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
            # nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，空洞率为6，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing=hebing(in_channel=dim_out*5)# 整合通道和空间特征的合并模块

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        larry=self.Hebing(feature_cat)
        larry_feature_cat=larry*feature_cat
        # 最终输出经过降维处理
        result = self.conv_cat(larry_feature_cat)

        return result

class encoder(nn.Module):
    def __init__(self,in_channel,output):
        super(encoder,self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channel, output, 3, padding=1),
            nn.BatchNorm2d(output),
            nn.PReLU()
        )
        self.mdfa = MDFA(in_channel, in_channel)
        # 使用不同尺度的自适应平均池化，并通过1x1卷积来减少特征维度
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(output, output, kernel_size=1),
            nn.BatchNorm2d(output),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Conv2d(output, output, kernel_size=1),
            nn.BatchNorm2d(output),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Conv2d(output, output, kernel_size=1),
            nn.BatchNorm2d(output),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Conv2d(output, output, kernel_size=1),
            nn.BatchNorm2d(output),
            nn.PReLU()
        )

        # 融合不同尺度的特征
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * output, output, kernel_size=1),
            nn.BatchNorm2d(output),
            nn.PReLU()
        )

    def forward(self, x):
        mdfa = self.mdfa(x)
        x = self.down_conv(mdfa)  # 降维
        conv1 = self.conv1(x)  # 1x1尺度
        conv2 = self.conv2(x)  # 2x2尺度
        conv3 = self.conv3(x)  # 3x3尺度
        conv4 = self.conv4(x)  # 6x6尺度

        # 将池化后的特征上采样到输入特征相同的尺寸，并进行融合
        conv1_up = F.interpolate(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv2_up = F.interpolate(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3_up = F.interpolate(conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv4_up = F.interpolate(conv4, size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))  # 在通道维度上进行拼接并通过1x1卷积融合



class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(SimpleEncoder, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )

    def forward(self, x1, x2):
        x = torch.concat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def kernel_size(in_channel):  # 计算一维卷积的核大小，利用的是ECA注意力中的参数[动态卷积核]
    k = int((math.log2(in_channel) + 1) // 2)
    return k + 1 if k % 2 == 0 else k


class MultiScaleFeatureExtractor(nn.Module):  # 多尺度特征提取器[对T1和T2不同时刻的特征进入到不同尺寸的卷积核加强提取]

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 使用不同尺寸的卷积核进行特征提取
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 分别使用不同尺寸的卷积核进行卷积操作
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out = out1 + out2 + out3  # 将不同尺度的特征相加
        return out


class ChannelAttention(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        # 使用自适应平均池化和最大池化提取全局信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        # 使用一维卷积计算通道注意力
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        # 对 t1 和 t2 进行平均池化和最大池化
        t1_channel_avg_pool = self.avg_pool(t1)
        t1_channel_max_pool = self.max_pool(t1)
        t2_channel_avg_pool = self.avg_pool(t2)
        t2_channel_max_pool = self.max_pool(t2)
        # 将池化结果拼接并转换维度
        channel_pool = torch.cat([
            t1_channel_avg_pool, t1_channel_max_pool,
            t2_channel_avg_pool, t2_channel_max_pool
        ], dim=2).squeeze(-1).transpose(1, 2)
        # 使用一维卷积计算通道注意力
        t1_channel_attention = self.channel_conv1(channel_pool)
        t2_channel_attention = self.channel_conv2(channel_pool)
        # 堆叠并使用Softmax进行归一化
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention], dim=0)
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)
        return channel_stack


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用二维卷积计算空间注意力
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        # 计算 t1 和 t2 的平均值和最大值
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]
        # 将平均值和最大值拼接
        spatial_pool = torch.cat([
            t1_spatial_avg_pool, t1_spatial_max_pool,
            t2_spatial_avg_pool, t2_spatial_max_pool
        ], dim=1)
        # 使用二维卷积计算空间注意力
        t1_spatial_attention = self.spatial_conv1(spatial_pool)
        t2_spatial_attention = self.spatial_conv2(spatial_pool)
        # 堆叠并使用Softmax进行归一化
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)
        spatial_stack = self.softmax(spatial_stack)
        return spatial_stack


class TFAM(nn.Module):
    """时序融合注意力模块"""

    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention()
    def forward(self, t1, t2):
        # 计算通道和空间注意力
        channel_stack = self.channel_attention(t1, t2)
        spatial_stack = self.spatial_attention(t1, t2)
        # 加权相加并进行融合
        stack_attention = channel_stack + spatial_stack + 1
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2
        return fuse


class BFM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.multi_scale_extractor = MultiScaleFeatureExtractor(in_channel, in_channel)
        self.tfam = TFAM(in_channel)

    def forward(self, t1, t2):
        # 进行多尺度特征提取
        t1_multi_scale = self.multi_scale_extractor(t1)
        t2_multi_scale = self.multi_scale_extractor(t2)
        # 使用TFAM进行融合
        output = self.tfam(t1_multi_scale, t2_multi_scale)
        return output


class CFFM(nn.Module):
    def __init__(self, channels, dim):
        super(CFFM, self).__init__()
        self.global_h1 = nn.AdaptiveMaxPool2d((None, 1))
        self.global_w1 = nn.AdaptiveMaxPool2d((1, None))

        self.global_h2 = nn.AdaptiveMaxPool2d((None, 1))
        self.global_w2 = nn.AdaptiveMaxPool2d((1, None))

        self.conv1 = SeparableConv2d(channels, dim, 3, 1, padding=2, dilation=2)
        self.conv2 = SeparableConv2d(channels, dim, 3, 1, padding=2, dilation=2)

        self.conv3 = nn.Conv2d(dim * 2 + 4, channels, 3, 1, padding=2, dilation=2)

        self.bn = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(channels)
        self.action = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x1, x2):
        change1 = self.action(self.bn(self.conv1(x1)))
        h1 = self.global_h1(change1)
        w1 = self.global_w1(change1)
        h1 = torch.transpose(h1, 1, 3)
        w1 = torch.transpose(w1, 1, 2)

        change2 = self.conv2(x2)
        h2 = self.global_h2(change2)
        w2 = self.global_w2(change2)
        h2 = torch.transpose(h2, 1, 3)
        w2 = torch.transpose(w2, 1, 2)

        all = torch.cat([change1, h1, w1, change2, h2, w2], dim=1)
        output = self.action(self.bn2(self.conv3(all)))

        return output


class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准


class ASPP(nn.Module):
    def __init__(self, dims, rate=[5,6,7]):
        super(ASPP, self).__init__()

        # self.pool = nn.MaxPool2d(2)
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                dims, dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=False),
            # nn.ReLU(),
            nn.BatchNorm2d(dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                dims, dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                dims, dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(dims),
        )

        self.de_ch = nn.Conv2d(len(rate) * dims, dims, 1)
        self._init_weights()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = self.pool(x)
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        concat = torch.cat([x1, x2, x3], dim=1)
        output = self.de_ch(concat)
        # output = self.pool(de_ch)

        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                m.weight.data = torch.ones_like(m.weight.data)
                m.bias.data = torch.zeros_like(m.bias.data)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet, self).__init__()
        # Encoder

        self.enc2_1 = encoder(in_channels, base_channels)
        self.enc2_2 = encoder(base_channels * 2, base_channels * 2)
        self.enc2_3 = encoder(base_channels * 4, base_channels * 4)
        self.enc2_4 = encoder(base_channels * 8, base_channels * 8)

        # self.downsample = downsample()
        self.wt = wt()
        self.iwt = iwt()

        self.enc1_1 = SimpleEncoder(in_channels, base_channels // 2, base_channels)
        self.enc1_2 = SimpleEncoder(base_channels * 2, 3 * base_channels // 2, base_channels * 2)
        self.enc1_3 = SimpleEncoder(base_channels * 4, base_channels * 3, base_channels * 4)
        self.enc1_4 = SimpleEncoder(base_channels * 8, base_channels * 6, base_channels * 8)

        self.cffm1 = CFFM(base_channels, dim=128)
        self.cffm2 = CFFM(base_channels * 2, dim=64)
        self.cffm3 = CFFM(base_channels * 4, dim=32)
        self.cffm4 = CFFM(base_channels * 8, dim=16)

        self.bottleneck = BFM(in_channel=base_channels * 16)
        # self.se_1 = SEAttention(channel=64)
        # self.se_2 = SEAttention(channel=128)
        # self.se_3 = SEAttention(channel=256)
        # self.se_4 = SEAttention(channel=512)

        # Bottleneck
        # self.bottleneck_conv = SeparableConv2d(in_channels=base_channels*8,out_channels=base_channels*16)
        self.aspp_1 = ASPP(dims=1024)
        self.aspp_2 = ASPP(dims=64)

        # Decoder
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)
        # self.up4 = up_InterpConv(channels=base_channels * 16)
        # self.up3 = up_InterpConv(channels=base_channels * 8)
        # self.up2 = up_InterpConv(channels=base_channels * 4)
        # self.up1 = up_InterpConv(channels=base_channels * 2)

        self.poola = nn.AvgPool2d(2, stride=2)

        # CFFM for skip connections

        # Final prediction
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_1 = self.enc1_1(x)
        enc2_1 = self.enc2_1(x)
        downsample1 = self.wt(enc2_1)
        cffm1 = self.cffm1(enc1_1, enc2_1)
        d_cffm1 = self.wt(cffm1)
        enc1_2 = self.enc1_2(d_cffm1)
        enc2_2 = self.enc2_2(downsample1)
        downsample2 = self.wt(enc2_2)
        cffm2 = self.cffm2(enc1_2, enc2_2)
        d_cffm2 = self.wt(cffm2)
        enc1_3 = self.enc1_3(d_cffm2)
        enc2_3 = self.enc2_3(downsample2)
        downsample3 = self.wt(enc2_3)
        cffm3 = self.cffm3(enc1_3, enc2_3)
        d_cffm3 = self.wt(cffm3)

        enc1_4 = self.enc1_4(d_cffm3)
        enc2_4 = self.enc2_4(downsample3)
        downsample4 = self.wt(enc2_4)
        cffm4 = self.cffm4(enc1_4, enc2_4)
        d_cffm4 = self.wt(cffm4)
        bottleneck = self.bottleneck(downsample4, d_cffm4)
        aspp_1 = self.aspp_1(bottleneck)

        up4 = self.iwt(aspp_1)
        # se_4 = self.se_4(cffm4)
        dec4 = self.dec4(cffm4, up4)
        up3 = self.iwt(dec4)
        # se_3 = self.se_3(cffm3)
        dec3 = self.dec3(cffm3, up3)
        up2 = self.iwt(dec3)
        # se_2 = self.se_2(cffm2)
        dec2 = self.dec2(cffm2, up2)
        up1 = self.iwt(dec2)
        # se_1 = self.se_1(cffm1)
        dec1 = self.dec1(cffm1, up1)

        out = self.final_conv(dec1)

        return out


if __name__ == '__main__':
    model = UNet(in_channels=1, out_channels=2)
    model = model.to("cuda")
    input_tensor = torch.randn(16, 1, 128, 128).to("cuda")  # Example input
    output = model(input_tensor)
    print(output.shape)  # Expected output: torch.Size([1, 2, 256, 256])