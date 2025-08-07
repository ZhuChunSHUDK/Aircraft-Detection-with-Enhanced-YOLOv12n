import torch
import torch.nn as nn
import torch.nn.functional as F

class SAC(nn.Module):
    """
    Switchable Atrous Convolution: applies parallel atrous convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, atrous_rates=[1, 3, 5], stride=1, bias=False):
        super().__init__()
        self.branches = nn.ModuleList()
        for rate in atrous_rates:
            self.branches.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=rate, dilation=rate, bias=bias)
            )
        self.conv1x1 = nn.Conv2d(len(atrous_rates) * out_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        concat = torch.cat(outs, dim=1)
        y = self.conv1x1(concat)
        return self.act(self.bn(y))


#task.py  --->  from ultralytics.nn.modules.sac_ema_dyhead import SAC

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if  m is SAC:
            c1 = ch[f]
            c2 = args[0]  # 输出通道
            args = [c1, c2, *args[1:]]  # c1, out_channels, kernel, atrous_rates
