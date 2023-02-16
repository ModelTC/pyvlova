# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from ..op import CombinedOp, ElementwiseAdd, Linear, ReLU, SequenceOp
from .utils import *


class BasicBlock(CombinedOp):
    expansion = 1

    def __init__(self, name, in_shape, out_channel, stride=1, downsample=None):
        in_shape = shape2d(in_shape)
        self.conv1 = conv(name + '.conv1', in_shape, out_channel, 3, stride, 1)
        self.relu1 = mock(ReLU, name + '.relu1', self.conv1)
        self.conv2 = conv(name + '.conv2', self.conv1, out_channel, 3, 1, 1)
        self.relu2 = mock(ReLU, name + '.relu2', self.conv2)
        self.eltwise_add = mock(
            ElementwiseAdd, name + '.eltwise_add', self.conv2
        )
        self.batch = self.relu2.batch
        self.out_channel = self.relu2.channel
        self.out_height = self.relu2.height
        self.out_width = self.relu2.width
        self.downsample = downsample
        self.stride = stride
        ops = [v for v in self.__dict__.values() if isinstance(v, BaseOp)]
        super().__init__(name=name, ops=ops)

    def calc(self, x):
        residual = x
        out = self.conv1.calc(x)
        out = self.relu1.calc(out)
        out = self.conv2.calc(out)
        if self.downsample is not None:
            residual = self.downsample.calc(x)
        out = self.eltwise_add.calc(out, residual)
        out = self.relu2.calc(out)
        return out


class Bottleneck(CombinedOp):
    expansion = 4

    def __init__(self, name, in_shape, out_channel, stride=1, downsample=None):
        self.conv1 = conv(name + '.conv1', in_shape, out_channel, 1)
        self.relu1 = mock(ReLU, name + '.relu1', self.conv1)
        self.conv2 = conv(
            name + '.conv2', self.conv1, out_channel, 3, stride, 1
        )
        self.relu2 = mock(ReLU, name + '.relu2', self.conv2)
        self.conv3 = conv(name + '.conv3', self.conv2, out_channel * 4, 1)
        self.relu3 = mock(ReLU, name + '.relu3', self.conv3)
        self.eltwise_add = mock(
            ElementwiseAdd, name + '.eltwise_add', self.conv3
        )
        self.batch = self.relu3.batch
        self.out_channel = self.relu3.channel
        self.out_height = self.relu3.height
        self.out_width = self.relu3.width
        self.downsample = downsample
        self.stride = stride
        ops = [v for v in self.__dict__.values() if isinstance(v, BaseOp)]
        super().__init__(name=name, ops=ops)

    def calc(self, x):
        residual = x
        out = self.conv1.calc(x)
        out = self.relu1.calc(out)
        out = self.conv2.calc(out)
        out = self.relu2.calc(out)
        out = self.conv3.calc(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.eltwise_add.calc(out, residual)
        out = self.relu3.calc(out)
        return out


class ResNet(CombinedOp):
    def __init__(
        self,
        name,
        in_shape,
        block,
        layers,
        num_classes=1000,
        deep_stem=False,
        avg_down=False,
    ):
        self.inplanes = 64
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        if self.deep_stem:
            conv1 = conv(name + '.stem.conv1', in_shape, 32, 3, 2, 1)
            relu1 = mock(ReLU, name + '.stem.relu1', conv1)
            conv2 = conv(name + '.stem.conv2', conv1, 32, 3, 1, 1)
            relu2 = mock(ReLU, name + '.stem.relu2', conv2)
            conv3 = conv(name + '.stem.conv3', conv2, 64, 3, 1, 1)
            self.conv1 = SequenceOp(
                name='.stem', ops=[conv1, relu1, conv2, relu2, conv3]
            )
        else:
            self.conv1 = conv(name + '.conv1', in_shape, 64, 7, 2, 3)
        self.relu1 = mock(ReLU, name + '.relu1', self.conv1)
        self.maxpool = pool(name + '.maxpool', self.relu1, 3, 2, 1, 'max')
        self.layer1 = self._make_layer(
            name + '.layer1', self.maxpool, block, 64, layers[0]
        )
        self.layer2 = self._make_layer(
            name + '.layer2', self.layer1, block, 128, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            name + '.layer3', self.layer2, block, 256, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            name + '.layer4', self.layer3, block, 512, layers[3], stride=2
        )
        self.avgpool = pool(name + '.avgpool', self.layer4, 7, 1, 0, 'avg')
        self.flatten = flatten2d(name + '.flatten', self.avgpool)
        self.fc = Linear(
            batch=self.flatten.batch,
            in_channel=512 * block.expansion,
            out_channel=num_classes,
            biased=True,
            name=name + '.linear',
        )
        ops = [v for v in self.__dict__.values() if isinstance(v, BaseOp)]
        super().__init__(name=name, ops=ops)

    def _make_layer(
        self, name, prev, block, planes, blocks, stride=1, avg_down=False
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                raise NotImplementedError()
                # downsample = nn.Sequential(
                #     nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                #     nn.Conv2d(self.inplanes, planes * block.expansion,
                #               kernel_size=1, stride=1, bias=False),
                #     BN(planes * block.expansion),
                # )
            else:
                downsample = conv(
                    name + '.downsample',
                    prev,
                    planes * block.expansion,
                    1,
                    stride,
                )

        layers = []
        layers.append(
            block(
                name + '.' + str(len(layers)), prev, planes, stride, downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(name + '.' + str(len(layers)), layers[-1], planes)
            )

        return SequenceOp(name=name, ops=layers)

    def calc(self, x):
        x = self.conv1.calc(x)
        x = self.relu1.calc(x)
        x = self.maxpool.calc(x)
        x = self.layer1.calc(x)
        x = self.layer2.calc(x)
        x = self.layer3.calc(x)
        x = self.layer4.calc(x)
        x = self.avgpool.calc(x)
        x = self.flatten.calc(x)
        x = self.fc.calc(x)
        return x


def resnet18(**kwargs):
    model = ResNet(
        'resnet18', [1, 3, 224, 224], BasicBlock, [2, 2, 2, 2], **kwargs
    )
    return model
