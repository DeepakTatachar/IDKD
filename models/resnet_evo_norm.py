"""
Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/ResNetEvoNorm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


@torch.jit.script
def instance_std(x, eps):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


@torch.jit.script
def group_std(x, eps):
    N, C, H, W = x.size()
    groups = 32
    groups = C if groups > C else groups
    x = x.view(N, groups, C // groups, H, W)
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.sqrt(var.add(eps)).view(N, C, H, W)


class EvoNorm2D(nn.Module):
    def __init__(
        self,
        input,
        non_linear=True,
        version="S0",
        efficient=True,
        affine=True,
        momentum=0.9,
        eps=1e-5,
        groups=32,
        training=True,
    ):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == "S0":
            self.swish = MemoryEfficientSwish()
        self.groups = groups
        self.eps = nn.Parameter(torch.FloatTensor([eps]), requires_grad=False)
        if self.version not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear and (
                (self.version == "S0" and not self.efficient) or self.version == "B0"
            ):
                self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
            self.register_buffer("v", None)
        self.register_buffer("running_var", torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == "S0":
            if self.non_linear:
                if not self.efficient:
                    num = x * torch.sigmoid(
                        self.v * x
                    )  # Original Swish Implementation, however memory intensive.
                else:
                    num = self.swish(
                        x
                    )  # Experimental Memory Efficient Variant of Swish
                return num / group_std(x, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == "B0":
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max(
                    (var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps)
                )
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta

class BasicBlockEvoNorm(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, version='S0'):
        super(BasicBlockEvoNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.evo_n1 = EvoNorm2D(planes, version=version)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.evo_n2 = EvoNorm2D(planes, version=version)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                EvoNorm2D(self.expansion * planes, version=version)
            )

    def forward(self, x):
        out = self.evo_n1(self.conv1(x))
        out = self.evo_n2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckEvoNorm(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, version='S0'):
        super(BottleneckEvoNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.evo_n1 = EvoNorm2D(planes, version=version)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.evo_n2 = EvoNorm2D(planes, version=version)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.evo_n2 = EvoNorm2D(self.expansion * planes, version=version)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                EvoNorm2D(self.expansion * planes, version=version)
            )

    def forward(self, x):
        out = self.evo_n1(self.conv1(x))
        out = self.evo_n2(self.conv2(out))
        out = self.evo_n2(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEvoNorm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, version='S0'):
        super(ResNetEvoNorm, self).__init__()
        self.in_planes = 64
        self.type = 'single'
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.evo_n1 = EvoNorm2D(64, version=version)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if(num_classes < 200):
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(2048 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.evo_n1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
      
        out = self.linear(out)
        return out


def ResNetEvoNorm18(num_classes=10):
    return ResNetEvoNorm(BasicBlockEvoNorm, [2, 2, 2, 2], num_classes=num_classes)


def ResNetEvoNorm34(num_classes=10):
    return ResNetEvoNorm(BasicBlockEvoNorm, [3, 4, 6, 3], num_classes=num_classes)


def ResNetEvoNorm50(num_classes=10):
    return ResNetEvoNorm(BottleneckEvoNorm, [3, 4, 6, 3], num_classes=num_classes)


def ResNetEvoNorm101(num_classes=10):
    return ResNetEvoNorm(BottleneckEvoNorm, [3, 4, 23, 3], num_classes=num_classes)


def ResNetEvoNorm152(num_classes=10):
    return ResNetEvoNorm(BottleneckEvoNorm, [3, 8, 36, 3], num_classes=num_classes)
