import torch
import torch.nn as nn
import torch.nn.functional as F

from models.awn import SliceableConv2d, SliceableLinear
from models.awn import SwitchableSharedBatchNorm2d
from models.awn import MaskTriangularConv2d
from models.awn import AWNet


class AWLeNet5(AWNet):
    def __init__(self, num_classes=10, input_size=28, conv2d=SliceableConv2d,
                 init_width_mult=1.0, slices=[1.0], divisor=1, min_channels=1):
        super(AWLeNet5, self).__init__()

        self.set_width_mult(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        n = self._slice(32, init_width_mult)
        if input_size == 28: # mnist
            inC = 1
            outL = 4
        elif input_size == 32: # cifar
            inC = 3
            outL = 5
        else:
            raise NotImplementedError('not support ' + dataset)

        log_slices = [0.25, 0.5, 0.75, 1.0]
        self.features = nn.Sequential(
            conv2d(inC, n, 3, 1, 0, fixed_in=True),
            SwitchableSharedBatchNorm2d(n, slices),
            nn.ReLU(),
            conv2d(n, n, 3, 1, 0),
            SwitchableSharedBatchNorm2d(n, slices),
            nn.ReLU(),
            nn.MaxPool2d(2),
            conv2d(n, n, 5, 1, 0),
            SwitchableSharedBatchNorm2d(n, slices),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            SliceableLinear(n*outL*outL, num_classes, fixed_out=True)
        )


def get_model(args):
    if 'TriaConv' in args.model_name:
        conv2d = MaskTriangularConv2d
    else:
        conv2d = SliceableConv2d

    return AWLeNet5(num_classes=args.num_classes,
                    input_size=args.image_size,
                    conv2d=conv2d,
                    init_width_mult=args.model_init_width_mult,
                    slices=args.model_width_mults)

