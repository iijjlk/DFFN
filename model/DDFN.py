import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def make_model(args, parent=False):
    return DDFN(in_chn=3, wf=10, depth=4)


class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=True):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=True):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Context(nn.Module):
    def __init__(self, in_channels=24, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        sa_x = self.conv_sa(input_x)
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = sa_x + ca_x
        return out


#
class IFM(nn.Module):
    def __init__(self, in_channels=24, kernel_size=3):
        super(IFM, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = Context(in_channels, kernel_size)
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, 1, 1, 0)
        self.kernel = nn.Conv2d(in_channels, in_channels * kernel_size * kernel_size, 1, 1, 0)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, pha, amp):
        fusion = self.fusion(torch.cat([x, pha, amp], dim=1))
        b, c, h, w = x.size()
        att = self.sekg(fusion)
        kers = self.kernel(att)
        filter_x = kers.reshape([b, c, self.kernel_size * self.kernel_size, h, w])

        unfold_x = self.unfold(x).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)

        return out + x



class IFAM(nn.Module):
    def __init__(self, wf=24, depth=4):
        super(IFAM, self).__init__()
        self.depth = depth
        self.wf = wf
        self.conv_amp = nn.ModuleList()
        self.conv_pha = nn.ModuleList()
        for i in range(depth - 1):
            self.conv_pha.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_amp.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))

        # for phase
        self.resize_pha = nn.ModuleList()
        self.resize_amp = nn.ModuleList()
        self.fusion_pha = nn.ModuleList()
        self.fusion_amp = nn.ModuleList()

        for i in range(self.depth - 1):
            self.resize_pha.append(nn.ModuleList())
            self.resize_amp.append(nn.ModuleList())

        for i in range(self.depth - 1):
            self.resize_pha[i] = nn.ModuleList()
            self.resize_amp[i] = nn.ModuleList()
            for j in range(self.depth - 1):
                if i < j:
                    self.resize_pha[i].append(
                        DownSample(in_channels=(2 ** i) * wf, scale_factor=2 ** (j - i), chan_factor=2, kernel_size=3))
                    self.resize_amp[i].append(
                        DownSample(in_channels=(2 ** i) * wf, scale_factor=2 ** (j - i), chan_factor=2, kernel_size=3))
                elif i == j:
                    self.resize_pha[i].append(None)
                    self.resize_amp[i].append(None)
                else:
                    self.resize_pha[i].append(
                        UpSample(in_channels=(2 ** i) * wf, scale_factor=2 ** (i - j), chan_factor=2, kernel_size=3))
                    self.resize_amp[i].append(
                        UpSample(in_channels=(2 ** i) * wf, scale_factor=2 ** (i - j), chan_factor=2, kernel_size=3))

            self.fusion_pha.append(nn.Conv2d((2 ** i) * wf * (depth - 1), (2 ** i) * wf, 1, 1, 0))
            self.fusion_amp.append(nn.Conv2d((2 ** i) * wf * (depth - 1), (2 ** i) * wf, 1, 1, 0))

    def forward(self, phas, amps):
        pha_feas = []
        amp_feas = []

        for i in range(self.depth - 1):
            pha_feas.append(self.conv_pha[i](phas[i]))
            amp_feas.append(self.conv_amp[i](amps[i]))

        for i in range(self.depth - 1):
            for j in range(self.depth - 1):
                if i != j:
                    x = torch.cat([pha_feas[i], self.resize_pha[j][i](pha_feas[j])], dim=1)
                    pha_feas[i] = x

                    y = torch.cat([amp_feas[i], self.resize_amp[j][i](amp_feas[j])], dim=1)
                    amp_feas[i] = y

            pha_feas[i] = self.fusion_pha[i](pha_feas[i])
            amp_feas[i] = self.fusion_amp[i](amp_feas[i])

        return pha_feas, amp_feas


class DDFN(nn.Module):
    def __init__(self, in_chn=3, wf=20, depth=4, relu_slope=0.2):
        super(DDFN, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        prev_channels = wf
        for i in range(depth):
            downsample = True if (i + 1) < depth else False
            # amplitude based residual block
            self.down_path_1.append(FFTConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope, use_FFT_AMP=True,
                                                 use_FFT_PHASE=False))
            # phase based residual block and use cross scale feature fusion (encs and decs from previous stage)
            self.down_path_2.append(
                FFTConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope, use_csff=downsample,
                             use_FFT_AMP=False, use_FFT_PHASE=True))
            prev_channels = (2 ** i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(
                UNetUpBlockStage1(prev_channels, (2 ** i) * wf, relu_slope, use_FFT_AMP=True, use_FFT_PHASE=False))
            self.up_path_2.append(
                UNetUpBlockStage2(prev_channels, (2 ** i) * wf, relu_slope, use_FFT_AMP=False, use_FFT_PHASE=True))
            self.skip_conv_1.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            prev_channels = (2 ** i) * wf

        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.fci = IFAM(wf, depth)

        self.last = nn.Conv2d(prev_channels, in_chn, 3, 1, 1, bias=True)

    def forward(self, x):
        image = x
        # stage 1 : amplitude  stage
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
        for i, up in enumerate(self.up_path_1):
            # print(x1.size(), "--", self.skip_conv_1[i](encs[-i-1]).size())
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        # stage 2 : phase  stage
        # ============================================================================================
        # ============================================================================================
        out_1_fft = torch.fft.rfft2(out_1, norm='backward')
        out_1_amp = torch.abs(out_1_fft)

        out_1_phase = torch.angle(out_1_fft)

        image_fft = torch.fft.rfft2(image, norm='backward')
        image_phase = torch.angle(image_fft)
        image_inverse = torch.fft.irfft2(out_1_amp * torch.exp(1j * image_phase), norm='backward')

        x2 = self.conv_02(image_inverse)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))  # 做消融
        # ============================================================================================
        # ============================================================================================
        phas = []
        for i, down in enumerate(self.down_path_2):  # UNet_2
            if (i + 1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i - 1])
                phas.append(x2_up)
            else:
                x2 = down(x2)

        decs.reverse()  # from high-resolution to low-resolution
        pha_feas, amp_feas = self.fci(phas, decs)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, pha_feas[-i - 1], amp_feas[-i - 1])

        out_2 = self.last(x2)
        out_2 = torch.sigmoid(out_2)
        out_2 = out_2 + image

        # return [out_1, out_1_amp, out_1_phase, out_2, image_phase, pha_feas, amp_feas]
        return [out_1, out_1_amp, out_1_phase, out_2, image_phase, phas, decs,image_inverse]


class FFTConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_FFT_PHASE=False,
                 use_FFT_AMP=False):
        super(FFTConvBlock, self).__init__()
        self.downsample = downsample

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff
        self.use_FFT_PHASE = use_FFT_PHASE
        self.use_FFT_AMP = use_FFT_AMP

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_fft_1 = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.relu_fft_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_fft_2 = nn.Conv2d(out_size, out_size, 1, 1, 0)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        if self.use_FFT_PHASE and self.use_FFT_AMP == False:
            x_res = self.identity(x)
            x_fft = torch.fft.fft2(x_res, dim=(-2, -1))
            x_amp = torch.abs(x_fft)
            x_phase = torch.angle(x_fft)

            x_phase = self.conv_fft_1(x_phase)
            x_phase = self.relu_fft_1(x_phase)
            x_phase = self.conv_fft_2(x_phase)
            x_fft_res = torch.fft.ifft2(x_amp * torch.exp(1j * x_phase), dim=(-2, -1))
            x_fft_res = x_fft_res.real

            out = out + x_res + x_fft_res
        elif self.use_FFT_AMP and self.use_FFT_PHASE == False:
            x_res = self.identity(x)
            x_fft = torch.fft.fft2(x_res, dim=(-2, -1))
            x_amp = torch.abs(x_fft)
            x_phase = torch.angle(x_fft)

            x_amp = self.conv_fft_1(x_amp)
            x_amp = self.relu_fft_1(x_amp)
            x_amp = self.conv_fft_2(x_amp)
            x_fft_res = torch.fft.ifft2(x_amp * torch.exp(1j * x_phase), dim=(-2, -1))
            x_fft_res = x_fft_res.real

            out = out + x_res + x_fft_res
        else:
            out = out + self.identity(x)

        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_size, out_size, bias=False):
    layer = nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetUpBlockStage1(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, use_FFT_AMP=False, use_FFT_PHASE=False):
        super(UNetUpBlockStage1, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = FFTConvBlock(in_size, out_size, False, relu_slope, use_FFT_AMP, use_FFT_PHASE)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class UNetUpBlockStage2(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, use_FFT_AMP=False, use_FFT_PHASE=False):
        super(UNetUpBlockStage2, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = FFTConvBlock(in_size, out_size, False, relu_slope, use_FFT_AMP, use_FFT_PHASE)

        self.afg = (IFM(out_size, 3))

    def forward(self, x, pha, amp):
        up = self.up(x)
        out = torch.cat([up, self.afg(up, pha, amp)], 1)
        out = self.conv_block(out)
        return out


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


import cv2
import numpy as np

def build():
    model = DDFN(in_chn=3, wf=20, depth=4)
    return model
if __name__ == "__main__":
    model = DDFN(in_chn=3, wf=30, depth=4).cuda()

    # img = cv2.imread("butterfly.bmp", cv2.IMREAD_UNCHANGED)
    # img = np.float32(img/255.)
    # x  = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
    # if x.shape[0] % 16 != 0 or x.shape[1] % 16 != 0:
    #     i = x.shape[0]
    #     j = x.shape[1]
    #     if x.shape[0] % 16 != 0:
    #         while i % 16 != 0:
    #             i-=1
    #     if x.shape[1]% 16 != 0:
    #         while j % 16 != 0:
    #             j -= 1
    # #     x = cv2.resize(x, (j,i))
    # height, width,_ = x.shape
    #
    # new_height = 2 ** int(np.ceil(np.log2(height)))
    # new_width = 2 ** int(np.ceil(np.log2(width)))
    #
    # # 计算需要填充的像素数量
    # pad_height = new_height - height
    # pad_width = new_width - width
    #
    # # 在图像周围添加填充
    # x_padded = cv2.copyMakeBorder(x, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT)
    #
    # # 在图像周围添加一圈像素
    # x = (np.asarray(x) / 255.0)
    #
    # x = torch.from_numpy(x).float().permute(2, 0, 1).cuda().unsqueeze(0)
    x = torch.rand(1, 3, 1072, 1920).cuda()
    from testFPS import FPSBenchmark
    from thop import profile

    with torch.no_grad():
        flops, params = profile(model, inputs=(x,))
        print('Params and FLOPs are {}M and {}G'.format(params / 1e6, flops / 1e9))
        #
        FPSBenchmark(
            model=model,
            input_size=(1, 3, 1072, 1920),
            device="cuda:0",
        ).repeat_measure_inference_speed()


