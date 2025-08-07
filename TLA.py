import torch
import torch.nn as nn

class TLA(nn.Module):
    """Tiny Layer Attention module (YOLO-TLA plugin)"""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # Channel attention mechanism (lightweight SE Block)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // 4, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        w = self.ca(x)
        return x * w


#task.py  --->  from ultralytics.nn.modules.func import ECA,CBAM, CoordAtt, GhostConv, BiFPNBlock, MobileViTBlock,TLA
