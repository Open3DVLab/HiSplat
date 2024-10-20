import torch
import torch.nn as nn
import torch.nn.functional as F

from .trident_conv import MultiScaleTridentConv


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, dilation=dilation, padding=dilation, stride=stride, bias=False
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        norm_layer=nn.InstanceNorm2d,
        num_output_scales=1,
        **kwargs,
    ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(
            feature_dims[2],
            stride=stride,
            norm_layer=norm_layer,
        )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(
                output_dim,
                output_dim,
                kernel_size=3,
                strides=strides,
                paddings=1,
                num_branch=self.num_branch,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/8 or 1/4

        x = self.conv2(x)

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out = [x]

        return out


class ResUnetEncoder(nn.Module):
    def __init__(
        self,
        norm_layer=nn.InstanceNorm2d,
        feature_dims=(32, 64, 128),
        **kwargs,
    ):
        super(ResUnetEncoder, self).__init__()
        # for the feature extract layer
        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=1, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.layer0 = self._make_layer(feature_dims[0], feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/1
        self.layer1 = self._make_layer(feature_dims[0], feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], feature_dims[2], stride=2, norm_layer=norm_layer)  # 1/4
        # the middle layers
        self.layer3 = self._make_layer(feature_dims[2], feature_dims[2], stride=1, norm_layer=norm_layer)  # 1/4
        self.conv2 = nn.Conv2d(feature_dims[2], feature_dims[2], 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, out_dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(in_dim, out_dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(out_dim, out_dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer0(x)
        feature_list.append(x)
        x = self.layer1(x)  # 1/2
        feature_list.append(x)
        x = self.layer2(x)  # 1/4
        feature_list.append(x)
        # the middle layer
        x = self.layer3(x)
        x = self.conv2(x)
        feature_list.append(x)
        return feature_list


class ResUnetDecoder(nn.Module):
    def __init__(
        self,
        norm_layer=nn.InstanceNorm2d,
        feature_dims=(32, 64, 128),
        **kwargs,
    ):
        super(ResUnetDecoder, self).__init__()

        # two decoders, up sample by 2x
        self.decode_layer1 = self._make_fine_layer(
            int(2 * feature_dims[-1]), feature_dims[-1], feature_dims[-2], stride=1, norm_layer=norm_layer
        )
        self.decode_layer0 = self._make_fine_layer(
            int(2 * feature_dims[-2]), feature_dims[-2], feature_dims[-3], stride=1, norm_layer=norm_layer
        )

        # the out layer
        self.out_layer0 = self._make_fine_layer(int(2 * feature_dims[0]), feature_dims[0], feature_dims[0])
        self.out_layer1 = self._make_fine_layer(int(2 * feature_dims[1]), feature_dims[1], feature_dims[1])
        self.out_layer2 = self._make_fine_layer(int(2 * feature_dims[2]), feature_dims[2], feature_dims[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_fine_layer(self, in_dim, mid_dim, out_dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(in_dim, mid_dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(mid_dim, out_dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def forward(self, feature_list: list[torch.Tensor], dino_feature_list=None):
        out_feature = []
        x = feature_list[-1]
        if dino_feature_list is not None:
            x = x + dino_feature_list[0]
        x = torch.cat([x, feature_list[-2]], dim=1)
        out_feature.append(self.out_layer2(x))
        x = self.decode_layer1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if dino_feature_list is not None:
            x = x + dino_feature_list[1]
        x = torch.cat([x, feature_list[-3]], dim=1)
        out_feature.append(self.out_layer1(x))
        x = self.decode_layer0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if dino_feature_list is not None:
            x = x + dino_feature_list[2]
        x = torch.cat([x, feature_list[-4]], dim=1)
        out_feature.append(self.out_layer0(x))
        return out_feature


class ResUnet(nn.Module):
    def __init__(
        self,
        norm_layer=nn.InstanceNorm2d,
        feature_dims=(32, 64, 128),
        **kwargs,
    ):
        super(ResUnet, self).__init__()
        self.encoder = ResUnetEncoder(norm_layer=norm_layer, feature_dims=feature_dims, **kwargs)
        self.decoder = ResUnetDecoder(norm_layer=norm_layer, feature_dims=feature_dims, **kwargs)
        self.up_dino_cnn = nn.ModuleList(
            [
                nn.Conv2d(64, feature_dims[-1], 1, bias=False),
                nn.Conv2d(64, feature_dims[-2], 1, bias=False),
                nn.Conv2d(64, feature_dims[-3], 1, bias=False),
            ]
        )

    def forward(self, x, dino_feature=None):
        feature_list = self.encoder(x)
        dino_feature_list = []
        for i in range(len(self.up_dino_cnn)):
            dino_feature_i = self.up_dino_cnn[i](
                F.interpolate(
                    dino_feature,
                    size=feature_list[len(feature_list) - i - 2].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            dino_feature_list.append(dino_feature_i)
        out_feature = self.decoder(feature_list, dino_feature_list)
        return out_feature
