import math
from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .dense import DenseBlock
from .duc import DenseUpsamplingConvolution
from .RepVGG import RepVGGBlock


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
class Attention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x
class PED(nn.Module):
    """
    Depth Filler Network (PED).
    """
    def __init__(self, in_channels = 4, hidden_channels = 64, L = 5, k = 12, use_DUC = True, **kwargs):
        super(PED, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC
        self.SFF=SqueezeAndExciteFusionAdd(channels_in=64)
        self.CA=Attention(channel=64)
        # First
        self.first = nn.Sequential(
            nn.Conv2d(4, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.first_r = nn.Sequential(
            nn.Conv2d(3, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.first_d = nn.Sequential(
            nn.Conv2d(1, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: skip
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        #self.dense1_conv2=Down_wt(in_ch=self.k,out_ch=self.hidden_channels)
        # Dense2: skip
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense2s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense2_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        #self.dense2_conv2 = Down_wt(in_ch=self.k, out_ch=self.hidden_channels)
        # Dense3: skip
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense3s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense3_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        #self.dense3_conv2 = Down_wt(in_ch=self.k, out_ch=self.hidden_channels)
        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense4_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        #self.dense4_conv2 = Down_wt(in_ch=self.k, out_ch=self.hidden_channels)
        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        #self.updense1 = RepVGGBlock(65, 65, kernel_size=3, stride=1, padding=1)
        self.updense1_duc = self._make_upconv(12, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 2
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        #self.updense2 = RepVGGBlock(65, 65, kernel_size=3, stride=1, padding=1)
        self.updense2_duc = self._make_upconv(12, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 3
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        #self.updense3 = RepVGGBlock(65, 65, kernel_size=3, stride=1, padding=1)
        self.updense3_duc = self._make_upconv(12, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 4
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        #self.updense4 = RepVGGBlock(65, 65, kernel_size=3, stride=1, padding=1)
        self.updense4_duc = self._make_upconv(12, self.hidden_channels, upscale_factor = 2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size = 1, stride = 1)
        )
    
    def _make_upconv(self, in_channels, out_channels, upscale_factor = 2):
        if self.use_DUC:
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor = upscale_factor)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = upscale_factor, stride = upscale_factor, padding = 0, output_padding = 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
    
    def forward(self, rgb, depth):
        # 720 x 1280 (rgb, depth) -> 360 x 640 (h)
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)
        # rgb=self.first_r(rgb)
        # depth_1=self.first_d(depth)
        # h=self.SFF(rgb,depth_1)
        # h=self.CA(h)
        h = self.first(torch.cat((rgb, depth), dim = 1))

        # dense1: 360 x 640 (h, depth1) -> 180 x 320 (h, depth2)
        depth1 = F.interpolate(depth, scale_factor = 0.5, mode = "nearest")
        # dense1: skip
        h_d1s = self.dense1s_conv1(h)
        h_d1s = self.dense1s(torch.cat((h_d1s, depth1), dim = 1))
        h_d1s = self.dense1s_conv2(h_d1s)
        # dense1: normal
        h = self.dense1_conv1(h)
        h = self.dense1(torch.cat((h, depth1), dim = 1))
        h = self.dense1_conv2(h)

        # dense2: 180 x 320 (h, depth2) -> 90 x 160 (h, depth3)
        depth2 = F.interpolate(depth1, scale_factor = 0.5, mode = "nearest")
        # dense2: skip
        h_d2s = self.dense2s_conv1(h)
        h_d2s = self.dense2s(torch.cat((h_d2s, depth2), dim = 1))
        h_d2s = self.dense2s_conv2(h_d2s)
        # dense2: normal
        h = self.dense2_conv1(h)
        h = self.dense2(torch.cat((h, depth2), dim = 1))
        h = self.dense2_conv2(h)

        # dense3: 90 x 160 (h, depth3) -> 45 x 80 (h, depth4)
        depth3 = F.interpolate(depth2, scale_factor = 0.5, mode = "nearest")
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        h_d3s = self.dense3s(torch.cat((h_d3s, depth3), dim = 1))
        h_d3s = self.dense3s_conv2(h_d3s)
        # dense3: normal
        h = self.dense3_conv1(h)
        h = self.dense3(torch.cat((h, depth3), dim = 1))
        h = self.dense3_conv2(h)

        # dense4: 45 x 80
        depth4 = F.interpolate(depth3, scale_factor = 0.5, mode = "nearest")
        h = self.dense4_conv1(h)
        h = self.dense4(torch.cat((h, depth4), dim = 1))
        h = self.dense4_conv2(h)

        # updense1: 45 x 80 -> 90 x 160
        h = self.updense1_conv(h)
        h = self.updense1(torch.cat((h, depth4), dim = 1))
        h = self.updense1_duc(h)

        # updense2: 90 x 160 -> 180 x 320
        h = torch.cat((h, h_d3s), dim = 1)
        h = self.updense2_conv(h)
        h = self.updense2(torch.cat((h, depth3), dim = 1))
        h = self.updense2_duc(h)

        # updense3: 180 x 320 -> 360 x 640
        h = torch.cat((h, h_d2s), dim = 1)
        h = self.updense3_conv(h)
        h = self.updense3(torch.cat((h, depth2), dim = 1))
        h = self.updense3_duc(h)

        # updense4: 360 x 640 -> 720 x 1280
        h = torch.cat((h, h_d1s), dim = 1)
        h = self.updense4_conv(h)
        h = self.updense4(torch.cat((h, depth1), dim = 1))
        h = self.updense4_duc(h)

        # final
        h = self.final(h)

        return rearrange(h, 'n 1 h w -> n h w')

