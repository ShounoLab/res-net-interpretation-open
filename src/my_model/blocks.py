import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def output_padding_size(H_out, H_in, kernel, stride, padding):
    return int(H_out - (H_in - 1) * stride + 2 * padding - kernel)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Block(nn.Sequential):
    pass


class MySequential(nn.Sequential):
    def forward(self, input, layer=None):
        prop_layer = None
        if layer is not None:
            index = layer.find(".")
            if index == -1:
                # k = int(layer)
                k = layer
                prop_layer = ""
            else:
                k = layer[:index]
                prop_layer = layer[index + len(".") :]

        # for i, module in enumerate(self._modules.values()):
        for key, module in self._modules.items():
            func = module.forward
            # get funciton argumnets
            argu = func.__code__.co_varnames[: func.__code__.co_argcount]
            if "layer" in argu:
                if layer is not None and key == k:
                    input = module(input, layer=prop_layer)
                    return input
                else:
                    input = module(input, layer=prop_layer)
            else:
                input = module(input)
        return input

    def rec(self, input):
        for i, module in enumerate(reversed(self._modules.values())):
            if isinstance(module, BasicBlock):
                input = module.rec(input)
            elif isinstance(module, nn.Conv2d):
                # TODO: monky patch. more cool
                # output_padding = H_out - (H_in - 1) * stride + 2 * padding - kernel
                kernel = module.weight.shape[-1]
                stride = module.stride[-1]
                padding = module.padding[-1]
                if stride == 1:
                    output_padding = 0
                elif stride == 2:
                    H = input.shape[-1]
                    output_padding = output_padding_size(
                        2 * H, H, kernel, stride, padding
                    )
                else:
                    raise ValueError(
                        "Un support {}. Need lower than 2".format(module.stride)
                    )

                input = F.conv_transpose2d(
                    input,
                    module.weight,
                    bias=module.bias,
                    stride=module.stride,
                    padding=module.padding,
                    output_padding=output_padding,
                )
            elif isinstance(module, nn.BatchNorm2d):
                var = module.running_var.clone().detach()
                mean = module.running_mean.clone().detach()
                bias = module.bias.clone().detach()
                weight = module.weight.clone().detach()
                input = F.batch_norm(input, bias, weight, var, mean, self.training)
            else:
                raise TypeError("method rec Un supported {}".format(type(module)))

        return input


class AddFunction(nn.Module):
    def __init__(self):
        super(AddFunction, self).__init__()

    def forward(self, x, y):
        return x + y


class BasicBlock(Block):

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        plain=False,
        projection=None,
        mode=None,
        bns=True,
    ):
        super(BasicBlock, self).__init__()
        _ = mode
        self.bns = bns
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.projection = projection
        self.downsample = downsample
        self.stride = stride
        self.add_func = AddFunction()
        self.isplain = plain

    def forward(self, x, layer=None):
        residual = x
        if layer is None:
            out = self.conv1(x)
            if self.bns:
                out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            if self.bns:
                out = self.bn2(out)

            if self.projection is not None:
                residual = self.projection(x)
            if self.downsample is not None:
                residual = self.downsample(residual)
            if not self.isplain:
                out = self.add_func(out, residual)
            out = self.relu2(out)

            return out
        else:
            out = self.conv1(x)
            if layer == "conv1":
                return out
            if self.bns:
                out = self.bn1(out)
            if layer == "bn1":
                return out
            out = self.relu1(out)
            if layer == "relu" or layer == "relu1":
                return out

            out = self.conv2(out)
            if layer == "conv2":
                return out
            if self.bns:
                out = self.bn2(out)
            if layer == "bn2":
                return out

            if self.projection is not None:
                residual = self.projection(x)
                if layer == "residual":
                    return residual
            if self.downsample is not None:
                index = layer.find(".")
                if index > -1:
                    prop_layer = layer[index + 1 :]
                    layer = layer[:index]
                    if layer != "downsample":
                        raise ValueError("No. {}".format(layer))
                else:
                    prop_layer = None

                residual = self.downsample(residual, prop_layer)
                if layer == "downsample":
                    return residual

            if not self.isplain:
                out = self.add_func(out, residual)
                if layer == "add_func":
                    return out
            elif layer == "add_func":
                return out

            out = self.relu2(out)
            return out

    def rec(self, x):
        out = torch.relu(x)
        # TODO: support CPU
        var = self.bn2.running_var.clone().detach()
        mean = self.bn2.running_mean.clone().detach()
        bias = self.bn2.bias.clone().detach()
        weight = self.bn2.weight.clone().detach()
        out = F.batch_norm(out, bias, weight, var, mean, self.training)
        out = F.conv_transpose2d(
            out,
            self.conv2.weight,
            bias=self.conv2.bias,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
        )
        out = torch.relu(out)
        var = self.bn1.running_var.clone().detach()
        mean = self.bn1.running_mean.clone().detach()
        bias = self.bn1.bias.clone().detach()
        weight = self.bn1.weight.clone().detach()
        out = F.batch_norm(out, bias, weight, var, mean, self.training)

        kernel = self.conv1.weight.shape[-1]
        stride = self.conv1.stride[-1]
        padding = self.conv1.padding[-1]
        if stride == 1:
            output_padding = 0
        elif stride == 2:
            H = out.shape[-1]
            output_padding = output_padding_size(2 * H, H, kernel, stride, padding)
        else:
            raise ValueError(
                "Un support {}. Need lower than 2".format(self.conv1.stride)
            )

        out = F.conv_transpose2d(
            out,
            self.conv1.weight,
            bias=self.conv1.bias,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            output_padding=output_padding,
        )
        residual = x
        if self.downsample is not None:
            residual = self.downsample.rec(x)

        if not self.isplain:
            out = self.add_func(out, residual)

        return out


class BasicBlockGeneral(Block):

    """
    General Basic Block
    """

    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, plain=False, mode=None
    ):
        super(BasicBlockGeneral, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.add_func = AddFunction()
        self.isplain = plain

        self.mode = mode

        if self.mode is None:
            pass
        elif self.mode == "0":
            # normal resnet
            # TODO: more cool skip_conv1 and skip_bn1
            if self.downsample is not None:
                self.skip_conv1 = self.downsample[0]
                self.skip_bn1 = self.downsample[1]
            else:
                self.skip_conv1 = nn.Identity()
                self.skip_bn1 = nn.Identity()
            self.skip_relu1 = nn.Identity()
            self.skip_conv2 = nn.Identity()
            self.skip_bn2 = nn.Identity()
        elif self.mode == "1":
            # split wide plain
            self.skip_conv1 = conv3x3(inplanes, planes, stride)
            self.skip_bn1 = nn.BatchNorm2d(planes)
            self.skip_relu1 = nn.ReLU(inplace=True)
            self.skip_conv2 = conv3x3(planes, planes)
            self.skip_bn2 = nn.BatchNorm2d(planes)
        elif self.mode == "2":
            # split wide plain without BN and ReLU
            # not trainable because of divergence
            self.skip_conv1 = conv3x3(inplanes, planes, stride)
            self.skip_bn1 = nn.Identity()
            self.skip_relu1 = nn.Identity()
            self.skip_conv2 = conv3x3(planes, planes)
            self.skip_bn2 = nn.Identity()
        elif self.mode == "3a":
            # no-cross projection
            if self.downsample is not None:
                self.skip_conv1 = self.downsample[0]
                self.skip_bn1 = self.downsample[1]
            else:
                self.skip_conv1 = nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    groups=inplanes,
                )
                self.skip_bn1 = nn.BatchNorm2d(planes)
            self.skip_relu1 = nn.ReLU(inplace=True)
            self.skip_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn2 = nn.BatchNorm2d(planes)
        elif self.mode == "3b":
            # no-cross projection
            self.skip_conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn1 = nn.BatchNorm2d(planes)
            self.skip_relu1 = nn.ReLU(inplace=True)
            self.skip_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn2 = nn.BatchNorm2d(planes)
        elif self.mode == "4a":
            # no-cross projection kernelsize=1
            if self.downsample is not None:
                self.skip_conv1 = self.downsample[0]
                self.skip_bn1 = self.downsample[1]
            else:
                self.skip_conv1 = nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    groups=inplanes,
                )
                self.skip_bn1 = nn.BatchNorm2d(planes)
            self.skip_relu1 = nn.ReLU(inplace=True)
            self.skip_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn2 = nn.BatchNorm2d(planes)
        elif self.mode == "4b":
            # no-cross projection kernelsize=1
            self.skip_conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn1 = nn.BatchNorm2d(planes)
            self.skip_relu1 = nn.ReLU(inplace=True)
            self.skip_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn2 = nn.BatchNorm2d(planes)

        elif self.mode == "5a":
            # no-cross projection kernelsize=1 without BN, ReLU
            if self.downsample is not None:
                self.skip_conv1 = self.downsample[0]
                self.skip_bn1 = self.downsample[1]
            else:
                self.skip_conv1 = nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    groups=inplanes,
                )
                self.skip_bn1 = nn.Identity()
            self.skip_relu1 = nn.Identity()
            self.skip_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn2 = nn.Identity()
        elif self.mode == "5b":
            # no-cross projection kernelsize=1 without BN, ReLU
            self.skip_conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn1 = nn.Identity()
            self.skip_relu1 = nn.Identity()
            self.skip_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                groups=inplanes,
            )
            self.skip_bn2 = nn.Identity()
        else:
            raise ValueError("Unknow mode {}".format(self.mode))

    def forward(self, x, layer=None):
        if layer is None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.mode is not None:
                skip_out = self.skip_conv1(x)
                skip_out = self.skip_bn1(skip_out)
                skip_out = self.skip_relu1(skip_out)
                skip_out = self.skip_conv2(skip_out)
                skip_out = self.skip_bn2(skip_out)
                residual = skip_out
            else:
                residual = x

            if not self.isplain:
                out = self.add_func(out, residual)
            out = self.relu2(out)

            return out
        else:
            import warnings

            warnings.warn("Unsupport layer_forward", UserWarning)

            out = self.conv1(x)
            if layer == "conv1":
                return out
            out = self.bn1(out)
            if layer == "bn1":
                return out
            out = self.relu1(out)
            if layer == "relu" or layer == "relu1":
                return out

            out = self.conv2(out)
            if layer == "conv2":
                return out
            out = self.bn2(out)
            if layer == "bn2":
                return out

            if self.projection is not None:
                residual = self.projection(x)
                if layer == "residual":
                    return residual
            if self.downsample is not None:
                index = layer.find(".")
                if index > -1:
                    prop_layer = layer[index + 1 :]
                    layer = layer[:index]
                    if layer != "downsample":
                        raise ValueError("No. {}".format(layer))
                else:
                    prop_layer = None

                residual = self.downsample(residual, prop_layer)

                if layer == "downsample":
                    return residual

            if not self.isplain:
                out = self.add_func(out, residual)
                if layer == "add_func":
                    return out

            out = self.relu2(out)
            return out


class Bottleneck(Block):

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        plain=False,
        mode=None,
        bns=True,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.add_func = AddFunction()
        self.isplain = plain

    def forward(self, x, layer=None):
        residual = x

        if layer is None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            if not self.isplain:
                out = self.add_func(out, residual)
            out = self.relu3(out)
        else:
            # out = x
            # for name, m in self.named_modules():
            #     out = m(out)
            #     if name == layer:
            #         return out

            out = self.conv1(x)
            if layer == "conv1":
                return out
            if self.bns:
                out = self.bn1(out)
            if layer == "bn1":
                return out
            out = self.relu1(out)
            if layer == "relu1":
                return out

            out = self.conv2(out)
            if layer == "conv2":
                return out
            if self.bns:
                out = self.bn2(out)
            if layer == "bn2":
                return out
            out = self.relu2(out)
            if layer == "relu2":
                return out

            out = self.conv3(out)
            if layer == "conv3":
                return out
            if self.bns:
                out = self.bn3(out)
            if layer == "bn3":
                return out

            if self.projection is not None:
                residual = self.projection(x)
                if layer == "residual":
                    return residual
            if self.downsample is not None:
                index = layer.find(".")
                if index > -1:
                    prop_layer = layer[index + 1 :]
                    layer = layer[:index]
                    if layer != "downsample":
                        raise ValueError("No. {}".format(layer))
                else:
                    prop_layer = None

                residual = self.downsample(residual, prop_layer)
                if layer == "downsample":
                    return residual

            if not self.isplain:
                out = self.add_func(out, residual)
                if layer == "add_func":
                    return out
            elif layer == "add_func":
                return out

            out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        plain=False,
        first_pad=True,
        block_mode=None,
        channel=64,
        ae=False,
        bns=True,
    ):
        self.ae = ae
        self.bns = bns
        if isinstance(channel, list):
            self.inplanes = channel[0]
            channel1 = channel[1]
            channel2 = channel[2]
            channel3 = channel[3]
            channel4 = channel[4]
        else:
            self.inplanes = channel
            channel1 = channel
            channel2 = int(channel * 2)
            channel3 = int(channel * 4)
            channel4 = int(channel * 8)

        super(ResNet, self).__init__()
        if first_pad:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=0, bias=False
            )
        self.block_mode = block_mode
        self.first_pad = first_pad
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.ae:
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, return_indices=True
            )
            self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
            # self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            # self.unpool = torch.ones(64, 1, 3, 3) / (3 * 3)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.unpool = None

        self.layer1 = self._make_layer(block, channel1, layers[0], plain=plain)
        self.layer2 = self._make_layer(
            block, channel2, layers[1], stride=2, plain=plain
        )
        self.layer3 = self._make_layer(
            block, channel3, layers[2], stride=2, plain=plain
        )
        self.layer4 = self._make_layer(
            block, channel4, layers[3], stride=2, plain=plain
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel4 * block.expansion, num_classes)
        self.tanh = nn.Tanh()

        for m in self.modules():
            ResNet.reset_weights(m)

    @classmethod
    def reset_weights(cls, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, plain=False):
        downsample = None
        # print(planes, stride, self.inplanes, block.expansion)
        if not plain:
            if stride != 1 or self.inplanes != planes * block.expansion:
                layers = [
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                ]
                if self.bns:
                    layers = layers + [nn.BatchNorm2d(planes * block.expansion)]
                downsample = MySequential(*layers)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                plain=plain,
                mode=self.block_mode,
                bns=self.bns,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    plain=plain,
                    mode=self.block_mode,
                    bns=self.bns,
                )
            )

        return MySequential(*layers)

    def forward(self, x, layers=None):
        if self.ae:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            unpool_size = x.shape[2:]
            x, indeces = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.layer4.rec(x)
            x = self.layer3.rec(x)
            x = self.layer2.rec(x)
            x = self.layer1.rec(x)
            x = self.unpool(x, indeces, output_size=unpool_size)

            x = self.relu(x)
            var = self.bn1.running_var.clone().detach()
            mean = self.bn1.running_mean.clone().detach()
            bias = self.bn1.bias.clone().detach()
            weight = self.bn1.weight.clone().detach()
            x = F.batch_norm(x, bias, weight, var, mean, self.training)

            kernel = self.conv1.weight.shape[-1]
            stride = self.conv1.stride[-1]
            padding = self.conv1.padding[-1]
            if stride == 1:
                output_padding = 0
            elif stride == 2:
                H = x.shape[-1]
                output_padding = output_padding_size(2 * H, H, kernel, stride, padding)
            else:
                raise ValueError(
                    "Un support {}. Need lower than 2".format(self.conv1.stride)
                )
            x = F.conv_transpose2d(
                x,
                self.conv1.weight,
                bias=self.conv1.bias,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                output_padding=output_padding,
            )
            x = self.tanh(x)
            return x

        if layers is None:
            x = self.conv1(x)
            if self.bns:
                x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # x = self.fc(x.view(x.size(0), -1))

            return x
        else:
            return self.layer_forward(x, layers=layers)

    def layer_forward(self, x, layers=None, breaking_point=None):
        acts = {}
        if layers is None:
            layers = ["fc"]
        if breaking_point is None:
            breaking_point = layers

        layer = layers[0]
        assert isinstance(layer, str)
        index = layer.find(".")
        if index > -1:
            prop_layer = layer[index + 1 :]
            layer = layer[:index]
        else:
            layer = None
            prop_layer = None

        x = self.conv1(x)
        if "conv1" in layers:
            acts["conv1"] = x
        if "conv1" in breaking_point:
            return acts

        if self.bns:
            x = self.bn1(x)
        if "bn1" in layers:
            acts["bn1"] = x
        if "bn1" in breaking_point:
            return acts

        x = self.relu(x)
        if "relu" in layers:
            acts["relu"] = x
        if "relu" in breaking_point:
            return acts

        x = self.maxpool(x)
        if "maxpool" in layers:
            acts["maxpool"] = x
        if "maxpool" in breaking_point:
            return acts

        if layer == "layer1":
            x = self.layer1(x, prop_layer)
            acts[layers[0]] = x
            return acts
        else:
            x = self.layer1(x)
        if "layer1" in layers:
            acts["layer1"] = x
        if "layer1" in breaking_point:
            return acts

        if layer == "layer2":
            x = self.layer2(x, prop_layer)
            acts[layers[0]] = x
            return acts
        else:
            x = self.layer2(x)
        if "layer2" in layers:
            acts["layer2"] = x
        if "layer2" in breaking_point:
            return acts

        if layer == "layer3":
            x = self.layer3(x, prop_layer)
            acts[layers[0]] = x
            return acts
        else:
            x = self.layer3(x)
        if "layer3" in layers:
            acts["layer3"] = x
        if "layer3" in breaking_point:
            return acts

        if layer == "layer4":
            x = self.layer4(x, prop_layer)
            acts[layers[0]] = x
            return acts
        else:
            x = self.layer4(x)
        if "layer4" in layers:
            acts["layer4"] = x
        if "layer4" in breaking_point:
            return acts

        x = self.avgpool(x)
        if "avgpool" in layers:
            acts["avgpool"] = x
        if "avgpool" in breaking_point:
            return acts

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if "fc" in layers:
            acts["fc"] = x
        if "fc" in breaking_point:
            return acts

        return acts
