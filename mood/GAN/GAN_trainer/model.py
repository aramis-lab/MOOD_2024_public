# %% Import

# numpy import
import numpy as np

# toch import for NN
import torch
from torch import nn

# %% ################## Generator -- U-net ##################

###################### Downsampling part ###################


class UNetDownBlock(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input block.
        out_size : (int) number of channels in the output block.

    """

    def __init__(self, in_size, out_size):
        super(UNetDownBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


###################### Upsampling part ###################


class UNetUpBlock(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input block.
        out_size : (int) number of channels in the output block.

    """

    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # skip connection
        x = self.model(x)
        return x


###################### Final block U-net ###################


class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


###################### Unet generator Assembly ###################


class Unet(nn.Module):
    """Descending part of the U-Net.

    Args:
        in_channels: (int) number of channels in the input image.
        out_channels : (int) number of channels in as input size for the FCN.

    """

    def __init__(self, in_channels=1, out_channels=1):
        super(Unet, self).__init__()

        self.down1 = UNetDownBlock(in_channels, 128)
        self.down2 = UNetDownBlock(128, 256)
        self.down3 = UNetDownBlock(256, 512)
        self.down4 = UNetDownBlock(512, 1024)
        self.down5 = UNetDownBlock(1024, 1024)

        self.up1 = UNetUpBlock(1024, 1024)
        self.up2 = UNetUpBlock(2048, 512)  # input channel *2 because of concatenation
        self.up3 = UNetUpBlock(1024, 256)
        self.up4 = UNetUpBlock(512, 128)

        self.final = FinalLayer(256, 1)

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)


def test_generator():
    x = torch.randn(1, 1, 128, 128, 128)
    model = Unet()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test_generator()


# %% ################## Discriminator ##################


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            DiscriminatorBlock(in_channels * 2, 64),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            nn.Conv3d(512, 1, 4, padding=0),
            nn.AvgPool3d(5),
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), dim=1)
        print(img_input.size())
        return self.model(img_input)


class Discriminator64(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator64, self).__init__()

        self.model = nn.Sequential(
            DiscriminatorBlock(in_channels * 2, 32),
            DiscriminatorBlock(32, 64),
            DiscriminatorBlock(64, 128),
            nn.Conv3d(128, 1, 4, padding=0),
            nn.AvgPool3d(5),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def test_discriminator():
    x1 = torch.randn(1, 1, 128, 128, 128)
    x2 = torch.randn(1, 1, 128, 128, 128)
    model = Discriminator()
    preds = model(x1, x2)
    print(preds.shape)


def test_discriminator_2():
    x1 = torch.randn(1, 1, 128, 128, 128)
    x2 = torch.randn(1, 1, 128, 128, 128)
    model = Discriminator()
    preds = model(x1, x2)
    print(preds.shape)


def test_discriminator_64():
    x1 = torch.randn(1, 1, 128, 128, 128)
    x2 = torch.randn(1, 1, 128, 128, 128)
    model = Discriminator()
    preds = model(x1, x2)
    print(preds.shape)


if __name__ == "__main__":
    test_discriminator()

# %% Attention Unet


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, norm_layer):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    """
    github: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py#L275
    """

    def __init__(self, img_ch=1, output_ch=1, norm_layer=nn.BatchNorm3d):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64, norm_layer=norm_layer)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, norm_layer=norm_layer)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, norm_layer=norm_layer)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, norm_layer=norm_layer)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, norm_layer=norm_layer)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, norm_layer=norm_layer)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256, norm_layer=norm_layer)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, norm_layer=norm_layer)

        self.Up4 = up_conv(ch_in=512, ch_out=256, norm_layer=norm_layer)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128, norm_layer=norm_layer)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, norm_layer=norm_layer)

        self.Up3 = up_conv(ch_in=256, ch_out=128, norm_layer=norm_layer)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64, norm_layer=norm_layer)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, norm_layer=norm_layer)

        self.Up2 = up_conv(ch_in=128, ch_out=64, norm_layer=norm_layer)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32, norm_layer=norm_layer)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, norm_layer=norm_layer)

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

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


# %%

import functools


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm3d,
        # norm_layer=nn.InstanceNorm3d,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
