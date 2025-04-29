import torch
import torch.nn as nn
import torch.nn.functional as F

class PreCM1(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 groups: int=1,
                 bias: int=0
                 ):
        super(PreCM1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        self.convtest = nn.Conv2d(in_channels // groups, out_channels, kernel_size, bias=False)


    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2)
        pl = int(prl // 2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        input = torch.cat([input,
                           torch.rot90(input, k=-1, dims=(2, 3)),
                           torch.rot90(input, k=-2, dims=(2, 3)),
                           torch.rot90(input, k=-3, dims=(2, 3))], dim=0)
        return F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, groups=self.groups)


class PreCM2(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: int,
        dilation:int = 1,
        groups: int=1
                 ):
        super(PreCM2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        weight_tensor0 = torch.Tensor(4 * out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor0)
        self.convtest = nn.Conv2d(in_channels // groups, out_channels * 4, kernel_size)


    def forward(self, input:torch.Tensor, output_shape:list):
        # assert len(list) == 2
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab//2)
        pl = int(prl//2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        out2 = F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, dilation=self.dilation, groups=self.groups)
        batch = b // 4
        oc = self.out_channels
        out2list = []
        for i in range(4):
            out2list.append(
                torch.rot90(out2[0 * batch: 0 * batch + batch, (i - 0) % 4 * oc: (i - 0) % 4 * oc + oc, :, :], k=(-i + 0) % 4, dims=(2, 3)) + \
                torch.rot90(out2[1 * batch: 1 * batch + batch, (i - 1) % 4 * oc: (i - 1) % 4 * oc + oc, :, :], k=(-i + 1) % 4, dims=(2, 3)) + \
                torch.rot90(out2[2 * batch: 2 * batch + batch, (i - 2) % 4 * oc: (i - 2) % 4 * oc + oc, :, :], k=(-i + 2) % 4, dims=(2, 3)) + \
                torch.rot90(out2[3 * batch: 3 * batch + batch, (i - 3) % 4 * oc: (i - 3) % 4 * oc + oc, :, :], k=(-i + 3) % 4, dims=(2, 3))
            )
        return torch.cat(out2list, dim=0)


class PreCM3(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int = 1,
                 bias: int = 0,
                 groups: int = 1
                 ):
        super(PreCM3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.groups = groups
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        self.convtest = nn.Conv2d(in_channels // groups, out_channels, kernel_size)

    def forward(self, input:torch.Tensor, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2)
        pl = int(prl // 2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        batch = b // 4
        out3 = F.conv2d(F.pad(input, padding), weight=self.convtest.weight, bias=None, stride=self.stride, groups=self.groups)
        return torch.rot90(out3[0 * batch: 1 * batch], k=0, dims=(2, 3)) + \
                torch.rot90(out3[1 * batch: 2 * batch], k=1, dims=(2, 3)) + \
                torch.rot90(out3[2 * batch: 3 * batch], k=2, dims=(2, 3)) + \
                torch.rot90(out3[3 * batch: 4 * batch], k=3, dims=(2, 3))
