import torch.nn as nn
import torch
import math
from torch.autograd import Variable

'''
ref:
[1] https://blog.csdn.net/weixin_45833008/article/details/108699461

'''
'''
1. Sequential 可以叠加 :model = Sequential(Sequential(m1,m2,m3),Sequential(m1,m2,m3))
2. 注意区分nn.model还是nn.function

TODO:
Add Width Multiplier 和 Resolution Multiplier.

'''


class MobileNetV1(nn.Module):
    def __init__(self,    num_class=1000):
        super(MobileNetV1, self).__init__()
        self.num_class = num_class
        self.config = [64, (128, 2), 128, (256, 2), 256, (512, 2),
                            512, 512, 512, 512, 512, (1024, 2), (1024, 2)]
        # num_filters,stride
        self.middle_layers = self._make_layers(32)

    def _make_layers(self, inp):
        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                                    nn.BatchNorm2d(oup),
                                    nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, groups=inp, bias=False),  # ?why bias=False
                                nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
                                nn.Conv2d(inp, outp, 3, stride=1,
                                          padding=0, bias=False),
                                nn.BatchNorm2d(oup),
                                nn.ReLU(inplace=True))

        layers = []
        for cfg in self.config:
            oup = cfg if isinstance(cfg, int) else cfg[0]
            stride = 1 if isinstance(cfg, int) else cfg[1]
            layers.append(conv_bn(inp, oup, stride))
            inp = oup
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.Conv2d(3, 32, 3, 2)(x)
        out = self.middle_layers(out)
        out = nn.AvgPool2d(7, padding=3)(out)
        out = out.view(x.size(0), -1)
        out = nn.Linear(1024, self.num_class)(out)
        out = nn.Softmax()(out)
        return out

if __name__ == '__main__':
    net = MobileNetV1()
    x = Variable(torch.zeros(1, 3, 32, 32)) # ref: https://blog.csdn.net/Houchaoqun_XMU/article/details/83009310
    # x = torch.zeros( 1, 3, 32, 32)
    y = net(x)
    print(y)
    # print(net)
