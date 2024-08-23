from mood.GAN.GAN_trainer.model import conv_block, Attention_block, up_conv
from torch.profiler import profile, ProfilerActivity
import torch
import torch.nn as nn
import time


class AttU_Net(nn.Module):
    """
    github: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py#L275
    """

    def __init__(self, img_ch=1, output_ch=1, norm_layer=nn.InstanceNorm3d):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16, norm_layer=norm_layer)
        self.Conv2 = conv_block(ch_in=16, ch_out=32, norm_layer=norm_layer)
        self.Conv3 = conv_block(ch_in=32, ch_out=64, norm_layer=norm_layer)
        self.Conv4 = conv_block(ch_in=64, ch_out=128, norm_layer=norm_layer)
        self.Conv5 = conv_block(ch_in=128, ch_out=256, norm_layer=norm_layer)

        self.Up5 = up_conv(ch_in=256, ch_out=128, norm_layer=norm_layer)
        self.Att5 = Attention_block(F_g=128, F_l=128, F_int=64, norm_layer=norm_layer)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128, norm_layer=norm_layer)

        self.Up4 = up_conv(ch_in=128, ch_out=64, norm_layer=norm_layer)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32, norm_layer=norm_layer)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64, norm_layer=norm_layer)

        self.Up3 = up_conv(ch_in=64, ch_out=32, norm_layer=norm_layer)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16, norm_layer=norm_layer)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32, norm_layer=norm_layer)

        self.Up2 = up_conv(ch_in=32, ch_out=16, norm_layer=norm_layer)
        self.Att2 = Attention_block(F_g=16, F_l=16, F_int=8, norm_layer=norm_layer)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16, norm_layer=norm_layer)

        self.Conv_1x1 = nn.Conv3d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_cpu(nn.Module):
    """
    github: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py#L275
    """

    def __init__(self, img_ch=1, output_ch=1, norm_layer=nn.InstanceNorm3d):
        super(AttU_Net_cpu, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16, norm_layer=norm_layer)
        self.Conv2 = conv_block(ch_in=16, ch_out=32, norm_layer=norm_layer)
        self.Conv3 = conv_block(ch_in=32, ch_out=64, norm_layer=norm_layer)
        self.Conv4 = conv_block(ch_in=64, ch_out=128, norm_layer=norm_layer)
        self.Conv5 = conv_block(ch_in=128, ch_out=256, norm_layer=norm_layer)

        self.Up5 = up_conv(ch_in=256, ch_out=128, norm_layer=norm_layer)
        self.Att5 = Attention_block(F_g=256, F_l=256, F_int=64, norm_layer=norm_layer)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128, norm_layer=norm_layer)

        self.Up4 = up_conv(ch_in=128, ch_out=64, norm_layer=norm_layer)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32, norm_layer=norm_layer)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64, norm_layer=norm_layer)

        self.Up3 = up_conv(ch_in=64, ch_out=32, norm_layer=norm_layer)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=32, norm_layer=norm_layer)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32, norm_layer=norm_layer)

        self.Up2 = up_conv(ch_in=32, ch_out=16, norm_layer=norm_layer)
        self.Att2 = Attention_block(F_g=32, F_l=32, F_int=16, norm_layer=norm_layer)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16, norm_layer=norm_layer)

        self.Conv_1x1 = nn.Conv3d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

if __name__ == "__main__":
    model = AttU_Net_cpu()
    inputs = torch.randn((1, 1, 256, 256, 256))
    criterion = nn.MSELoss()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        start_time = time.time()
        with torch.autocast(device_type="cpu"):
            outputs = model(inputs)
            loss = criterion(inputs, outputs)
        loss.backward()
        end_time = time.time() - start_time
    print(end_time)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=30))
