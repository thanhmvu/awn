import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings


def make_divisible(v, divisor=8, min_value=1):
    """
    Forked from slim: https://github.com/tensorflow/models/blob/ ...
        ... 0344c5503ee55e24f0de7f37336a6e08f10976fd/ ...
        ... research/slim/nets/mobilenet/mobilenet.py#L62-L69

    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Slimmable(nn.Module):

    def set_width_mult(self, slice): # call before forward() to set active width
        self.assert_slice(slice)
        self.curr_slice = slice

    def set_fixed_in_channels(self, fixed=True):
        self.fixed_in_channels = fixed

    def set_fixed_out_channels(self, fixed=True):
        self.fixed_out_channels = fixed

    def set_min_channels(self, min_channels=1):
        self.min_channels = min_channels

    def set_divisor(self, divisor=1):
        self.divisor = divisor

    def _init_slimmable(self, slice, fixed_in=False, fixed_out=False,
                        divisor=1, min_channels=1):
        self.set_width_mult(1.0)
        self.set_fixed_in_channels(fixed_in)
        self.set_fixed_out_channels(fixed_out)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

    def _slice(self, channels, slice=None):
        if slice is None:
            slice = self.curr_slice
        return make_divisible(round(slice * channels),
                              self.divisor, self.min_channels)

    @staticmethod
    def assert_slice(slice):
        assert(slice > 0.0 and slice <= 1.0)


class SliceableConv2d(nn.Conv2d, Slimmable):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 fixed_in=False, fixed_out=False, divisor=1, min_channels=1):

        super(SliceableConv2d, self).__init__(in_channels, out_channels,
                                              kernel_size, stride, padding,
                                              dilation, groups, bias)

        self.depthwise = (groups == in_channels)
        if groups > 1 and not self.depthwise:
            raise NotImplementedError

        self._init_slimmable(1.0, fixed_in, fixed_out, divisor, min_channels)

    def forward(self, input):
        base_out_channels, base_in_channels = self.weight.shape[:2]
        sliced_out_channels = (base_out_channels if self.fixed_out_channels
                               else self._slice(base_out_channels))
        sliced_in_channels = input.shape[1]

        sliced_weight = self.weight[
            :sliced_out_channels, :sliced_in_channels, :, :]
        sliced_bias = (None if self.bias is None
                       else self.bias[:sliced_out_channels])

        if self.depthwise:
            self.groups = sliced_in_channels

        out = F.conv2d(input, sliced_weight, sliced_bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out


class SliceableLinear(nn.Linear, Slimmable):
    def __init__(self, in_features, out_features, bias=True,
                 fixed_in=False, fixed_out=False, divisor=1, min_channels=1):
        super(SliceableLinear, self).__init__(in_features, out_features, bias)
        self._init_slimmable(1.0, fixed_in, fixed_out, divisor, min_channels)

    def forward(self, input):
        base_out_channels, base_in_channels = self.weight.shape
        sliced_in_channels = (base_in_channels if self.fixed_in_channels
                              else input.shape[1])
        sliced_out_channels = (base_out_channels if self.fixed_out_channels
                               else self._slice(base_out_channels))

        sliced_weight = self.weight[:sliced_out_channels, :sliced_in_channels]
        sliced_bias = (None if self.bias is None
                       else self.bias[:sliced_out_channels])

        out = F.linear(input, sliced_weight, sliced_bias)
        return out


class SharedBatchNorm2d(nn.BatchNorm2d, Slimmable):
    """ One single BatchNorm module that is sliceable """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, fixed_in=False, fixed_out=False,
                 divisor=1, min_channels=1):
        super(SharedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self._init_slimmable(1.0, fixed_in, fixed_out, divisor, min_channels)

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = (
                        1.0 / float(self.num_batches_tracked))
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        sliced_channels = self._slice(len(self.weight))
        assert(abs(sliced_channels - input.shape[1]) <= self.divisor), (
            'expect: {}, got: {}'.format(sliced_channels, input.shape[1]))
        sliced_channels = input.shape[1]

        sliced_weight = sliced_bias = None
        sliced_running_mean = sliced_running_var = None
        if self.affine:
            sliced_weight = self.weight[:sliced_channels]
            sliced_bias = self.bias[:sliced_channels]
        if self.track_running_stats:
            sliced_running_mean = self.running_mean[:sliced_channels]
            sliced_running_var = self.running_var[:sliced_channels]

        out = F.batch_norm(
            input, sliced_running_mean, sliced_running_var, sliced_weight,
            sliced_bias, self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        return out


class SwitchableSharedBatchNorm2d(Slimmable):
    """ Mix of switchable and shared BatchNorm """

    def __init__(self, num_features, slices, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 fixed_in=False, fixed_out=False, divisor=1, min_channels=1):
        super(SwitchableSharedBatchNorm2d, self).__init__()
        self.set_fixed_in_channels(fixed_in)
        self.set_fixed_out_channels(fixed_out)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        self.slices = slices
        self.bn_dict = nn.ModuleDict()
        for slice in self.slices:
            self.bn_dict[self.to_mdict_key(slice)] = SharedBatchNorm2d(
                self._slice(num_features, slice), eps, momentum, affine,
                track_running_stats, divisor, min_channels)
        self.set_width_mult(1.0)

    def to_mdict_key(self, in_float):
        return str(in_float).replace('.', '_')

    def to_bn_key(self, slice):
        return min([s for s in self.slices if s >= slice])

    def set_width_mult(self, slice):  # call before forward() to set active width
        self.assert_slice(slice)
        self.curr_slice = slice
        self.curr_bn_key = self.to_bn_key(self.curr_slice)
        self.curr_bn = self.bn_dict[self.to_mdict_key(self.curr_bn_key)]
        self.curr_bn.set_width_mult(self.curr_slice / self.curr_bn_key)

    def forward(self, x):
        return self.curr_bn(x)


class MaskTriangularConv2d(SliceableConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 fixed_in=False, fixed_out=False, divisor=1, min_channels=1,
                 min_width_mult=1e-6, min_sampled_width_mults=1000):
        super(MaskTriangularConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, fixed_in, fixed_out, divisor, min_channels)

        if self.depthwise:
            warnings.warn('For depthwise convolution, standard pytorch conv2d'
                          ' should be used instead of triangular conv2d')

        self.min_sampled_width_mults = min_sampled_width_mults
        self.generate_mask(min_width_mult)
        return

    def generate_mask(self, min_width_mult):
        self.assert_slice(min_width_mult)
        self.min_width_mult = min_width_mult
        self.mask = self._make_mask()
        return

    def _slice_weights(self, width_mult):
        out_channels, in_channels = self.weight.shape[:2]
        sliced_in_channels = (in_channels if self.fixed_in_channels
                              else self._slice(in_channels, width_mult))
        sliced_out_channels = (out_channels if self.fixed_out_channels
                               else self._slice(out_channels, width_mult))
        return (sliced_in_channels, sliced_out_channels)

    def _make_mask(self):
        C_out, C_in, H, W = self.weight.shape
        mask = torch.zeros([C_out, C_in, H, W])

        max_w = 1.0
        min_w = self.min_width_mult
        min_C_out = (C_out if self.fixed_out_channels
                     else self._slice(C_out, min_w))
        min_C_in = (C_in if self.fixed_in_channels
                    else self._slice(C_in, min_w))

        mask[:min_C_out, :min_C_in, :, :] = 1.0
        last_C_out = min_C_out

        max_C = max(C_out, C_in, self.min_sampled_width_mults)
        width_mults = sorted([min_w, max_w]
            + list(np.random.uniform(min_w, max_w, max_C)))
        for wm in width_mults:
            new_C_in, new_C_out = self._slice_weights(wm)
            mask[last_C_out:new_C_out, :new_C_in, :, :] = 1.0
            last_C_out = new_C_out

        mask.requires_grad_(False)
        self.num_active_weights = mask.sum().item()
        return mask

    def forward(self, input):
        with torch.no_grad():
            if self.mask.device != self.weight.device:
                self.mask = self.mask.to(self.weight.device)
            self.weight.data.mul_(self.mask)

        _, sliced_out_channels = self._slice_weights(self.curr_slice)
        sliced_in_channels = input.shape[1]

        sliced_weight = self.weight[
            :sliced_out_channels, :sliced_in_channels, :, :]
        sliced_bias = (None if self.bias is None
                       else self.bias[:sliced_out_channels])

        if self.depthwise:
            self.groups = sliced_in_channels

        out = F.conv2d(input, sliced_weight, sliced_bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out


class AWNet(Slimmable):
    '''
    Modules inherited from this class should
    include the following in their init method
        self.features = nn.Sequential(...)
        self.classifier = nn.Sequential(...)

        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        self.set_width_mult(1.0)
        self.reset_parameters()

    '''

    def set_width_mult(self, slice): # call before forward() to set active width
        self.assert_slice(slice)
        self.curr_slice = slice

        def _set_width_mult(layer):
            if isinstance(layer, Slimmable) and layer is not self:
                layer.set_width_mult(self.curr_slice)
        self.apply(_set_width_mult)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

