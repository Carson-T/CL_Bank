import math
from typing import List
import timm
import torch
import torch.nn as nn
from torchvision import models
from safetensors.torch import load_file
from timm.models import register_model
import torch.nn.functional as F
import types


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.last = last

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + basicblock
        if not self.last:
            out = F.relu(out, inplace=True)

        return out


class CifarResNet_v1(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3):
        super(CifarResNet_v1, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.num_features = 64 * block.expansion
        self.fc = nn.Linear(64*block.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)

        return x_3

    def forward_head(self, x, pre_logits=False):
        pooled = self.avgpool(x)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        return features if pre_logits else self.fc(features)

    def forward(self, x):
        feature_map = self.forward_features(x)
        output = self.forward_head(feature_map)

        return output

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


class Cifar_CosineResnet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3):
        super(Cifar_CosineResnet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8)
        self.num_features = 64 * block.expansion
        # self.fc = CosineLinear(64*block.expansion, 10)
        self.fc = nn.Linear(64*block.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)  # DownsampleA => DownsampleB

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)

        return x_3

    def forward_head(self, x, pre_logits=False):
        pooled = self.avgpool(x)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        return features if pre_logits else self.fc(features)

    def forward(self, x):
        feature_map = self.forward_features(x)
        output = self.forward_head(feature_map)

        return output

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


class CifarResNet_v2(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """
    def __init__(self, block: ResNetBasicblock, num_blocks: List[int],
                 num_classes: int, nf: int):
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(CifarResNet_v2, self).__init__()
        self.inplanes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.num_features = nf * 8 * block.expansion
        self.conv1 = nn.Conv2d(3, nf * 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.fc = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block: ResNetBasicblock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)
            layers.append(block(self.inplanes, planes, stride, downsample=downsample))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4

        return out

    def forward_head(self, x, pre_logits=False):
        out = F.avg_pool2d(x, x.shape[2])  # -> 512, 1, 1
        features = out.view(out.size(0), -1)  # 512

        return features if pre_logits else self.fc(features)

    def forward(self, x) -> torch.Tensor:
        feature_map = self.forward_features(x)
        output = self.forward_head(feature_map)

        return output

    @property
    def last_conv(self):
        return self.layer4[-1].conv_b

@register_model
def timm_resnet18(**kwargs):
    model = timm.create_model("resnet18", pretrained=False)
    return model

@register_model
def tv_resnet18(**kwargs):
    model = models.resnet18(weights=None)
    setattr(model, "num_features", model.fc.in_features)
    model.fc = nn.Identity()
    return model

@register_model
def mnist_resnet20(**kwargs):
    """Constructs a ResNet-20 model for MNIST."""
    model = CifarResNet_v1(ResNetBasicblock, 20, 1)
    return model

@register_model
def mnist_resnet32(**kwargs):
    """Constructs a ResNet-32 model for MNIST."""
    model = CifarResNet_v1(ResNetBasicblock, 32, 1)
    return model

@register_model
def cifar_resnet18(nclasses=10, nf=64, **kwargs):
    model = CifarResNet_v2(block=ResNetBasicblock, num_blocks=[2, 2, 2, 2], num_classes=nclasses, nf=nf)
    return model

@register_model
def cifar_resnet20(**kwargs):
    """Constructs a ResNet-20 model for CIFAR-10."""
    model = CifarResNet_v1(ResNetBasicblock, 20)
    return model

@register_model
def cifar_resnet32(**kwargs):
    """Constructs a ResNet-32 model for CIFAR-10."""
    model = CifarResNet_v1(ResNetBasicblock, 32)
    return model

@register_model
def cifar_resnet44(**kwargs):
    """Constructs a ResNet-44 model for CIFAR-10."""
    model = CifarResNet_v1(ResNetBasicblock, 44)
    return model

@register_model
def cifar_resnet56(**kwargs):
    """Constructs a ResNet-56 model for CIFAR-10."""
    model = CifarResNet_v1(ResNetBasicblock, 56)
    return model

@register_model
def cifar_resnet110(**kwargs):
    """Constructs a ResNet-110 model for CIFAR-10."""
    model = CifarResNet_v1(ResNetBasicblock, 110)
    return model

@register_model
def cifar_CosineResnet32(**kwargs):
    """Constructs a ResNet-32 model for UCIR CIFAR-10."""
    model = Cifar_CosineResnet(ResNetBasicblock, 32)
    return model
