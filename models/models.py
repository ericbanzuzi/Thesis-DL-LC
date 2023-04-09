import torch.nn as nn
from torchsummary import summary


class Conv2Plus1DFirst(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64)
        )


class Conv2Plus1DFirst_simple(nn.Sequential):
    def __init__(self):
        Mi = int((3 * 3 * 3 * 3 * 16) / (3 * 3 * 3 + 3 * 16))
        super().__init__(
            nn.Conv3d(3, Mi, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(Mi),
            nn.ReLU(inplace=True),
            nn.Conv3d(Mi, 16, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(16)
        )


class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2Plus1D, self).__init__()
        Mi = int((in_channels*3*3*3*out_channels)/(3*3*in_channels+3*out_channels))
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, Mi, kernel_size=(1, kernel_size[1], kernel_size[2]),
                      stride=(1, stride, stride), padding=(0, 1, 1)),
            nn.BatchNorm3d(Mi),
            nn.ReLU(inplace=True),
            nn.Conv3d(Mi, out_channels, kernel_size=(kernel_size[0], 1, 1),
                      stride=(stride, 1, 1), padding=(1, 0, 0))
        )

    def forward(self, x):
        return self.seq(x)


class Conv2Plus1DResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample):
        super(Conv2Plus1DResidualBlock2, self).__init__()
        if downsample:
            self.seq = nn.Sequential(
                # perform down sampling with (2+1)D max pooling
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                # conv block
                Conv2Plus1D(in_channels, out_channels, kernel_size, stride=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                Conv2Plus1D(out_channels, out_channels, kernel_size, stride=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.seq = nn.Sequential(
                Conv2Plus1D(in_channels, out_channels, kernel_size, stride=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                Conv2Plus1D(out_channels, out_channels, kernel_size, stride=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        return self.seq(x)


class Conv2Plus1DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super(Conv2Plus1DResidualBlock, self).__init__()
        self.downsample_flag = downsample
        if downsample:
            self.downsample = nn.Sequential(
                # perform down sampling with a 3D convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )

        self.seq = nn.Sequential(
            Conv2Plus1D(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            Conv2Plus1D(out_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.seq(x)
        if self.downsample_flag:
            residual = self.downsample(residual)
        out += residual
        return out


class Conv3DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample):
        super(Conv3DResidualBlock, self).__init__()
        self.downsample_flag = downsample
        if downsample:
            self.downsample = nn.Sequential(
                # perform down sampling with a 3D convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )

        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.seq(x)
        if self.downsample_flag:
            residual = self.downsample(residual)
        out += residual
        return out


class Conv2DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample):
        super(Conv2DResidualBlock, self).__init__()
        self.downsample_flag = downsample
        if downsample:
            self.downsample = nn.Sequential(
                # perform down sampling with a 3D convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride)),
                nn.BatchNorm3d(out_channels)
            )

        self.seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=(1, stride, stride), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.seq(x)
        if self.downsample_flag:
            residual = self.downsample(residual)
        out += residual
        return out


class R2Plus1D(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1D, self).__init__()
        self.conv1 = Conv2Plus1DFirst()
        self.relu1 = nn.ReLU(inplace=True)
        # First 2+1D block
        self.conv2_1 = Conv2Plus1DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = Conv2Plus1DResidualBlock(64, 64, (3, 3, 3), 1, False)
        self.relu2_2 = nn.ReLU(inplace=True)
        # Second 2+1D block
        self.conv3_1 = Conv2Plus1DResidualBlock(64, 128, (3, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = Conv2Plus1DResidualBlock(128, 128, (3, 3, 3), 1, False)
        self.relu3_2 = nn.ReLU(inplace=True)
        # Third 2+1D block
        self.conv4_1 = Conv2Plus1DResidualBlock(128, 256, (3, 3, 3), 2, True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = Conv2Plus1DResidualBlock(256, 256, (3, 3, 3), 1, False)
        self.relu4_2 = nn.ReLU(inplace=True)
        # Fourth 2+1D block
        self.conv5_1 = Conv2Plus1DResidualBlock(256, 512, (3, 3, 3), 2, True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = Conv2Plus1DResidualBlock(512, 512, (3, 3, 3), 1, False)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MC3_18(nn.Module):
    def __init__(self, num_classes):
        super(MC3_18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64)
        )
        self.relu1 = nn.ReLU(inplace=True)
        # First 2+1D block
        self.conv2_1 = Conv3DResidualBlock(64, 64, (3, 3, 3), False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = Conv3DResidualBlock(64, 64, (3, 3, 3), False)
        self.relu2_2 = nn.ReLU(inplace=True)
        # Second 2+1D block
        self.conv3_1 = Conv2DResidualBlock(64, 128, (1, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = Conv2DResidualBlock(128, 128, (1, 3, 3), 1, False)
        self.relu3_2 = nn.ReLU(inplace=True)
        # Third 2+1D block
        self.conv4_1 = Conv2DResidualBlock(128, 256, (1, 3, 3), 2, True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = Conv2DResidualBlock(256, 256, (1, 3, 3), 1, False)
        self.relu4_2 = nn.ReLU(inplace=True)
        # Fourth 2+1D block
        self.conv5_1 = Conv2DResidualBlock(256, 512, (1, 3, 3), 2, True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = Conv2DResidualBlock(512, 512, (1, 3, 3), 1, False)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MC3_18_simple(nn.Module):
    def __init__(self, num_classes):
        super(MC3_18_simple, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64)
        )
        self.relu1 = nn.ReLU(inplace=True)
        # First 2+1D block
        self.conv2_1 = Conv3DResidualBlock(64, 64, (3, 3, 3), False)
        self.relu2_1 = nn.ReLU(inplace=True)
        # Second 2+1D block
        self.conv3_1 = Conv2DResidualBlock(64, 128, (1, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class R2Plus1D_simple(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1D_simple, self).__init__()
        self.conv1 = Conv2Plus1DFirst_simple()
        self.relu1 = nn.ReLU(inplace=True)
        # First 2+1D block
        self.conv2_1 = Conv2Plus1DResidualBlock(16, 16, (3, 3, 3), 1, False)
        self.relu2_1 = nn.ReLU(inplace=True)
        # self.conv2_2 = Conv2Plus1DResidualBlock(32, 32, (3, 3, 3), 1, False)
        # self.relu2_2 = nn.ReLU(inplace=True)
        # Second 2+1D block
        self.conv3_1 = Conv2Plus1DResidualBlock(16, 32, (3, 3, 3), 2, True)
        self.relu3_1 = nn.ReLU(inplace=True)
        # self.conv3_2 = Conv2Plus1DResidualBlock(128, 128, (3, 3, 3), 1, False)
        # self.relu3_2 = nn.ReLU(inplace=True)
        # Third 2+1D block
        self.conv4_1 = Conv2Plus1DResidualBlock(32, 64, (3, 3, 3), 2, True)
        self.relu4_1 = nn.ReLU(inplace=True)
        # self.conv4_2 = Conv2Plus1DResidualBlock(256, 256, (3, 3, 3), 1, False)
        # self.relu4_2 = nn.ReLU(inplace=True)
        # Fourth 2+1D block
        self.conv5_1 = Conv2Plus1DResidualBlock(64, 128, (3, 3, 3), 2, True)
        self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = Conv2Plus1DResidualBlock(512, 512, (3, 3, 3), 1, False)
        # self.relu5_2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Final fully connected layer
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        # x = self.conv2_2(x)
        # x = self.relu2_2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        # x = self.conv3_2(x)
        # x = self.relu3_2(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        # x = self.conv4_2(x)
        # x = self.relu4_2(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        # x = self.conv5_2(x)
        # x = self.relu5_2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


from torchvision.models.video import r2plus1d_18, R3D_18_Weights, mc3_18

if __name__ == '__main__':
    model = R2Plus1D(num_classes=3)
    print(summary(model, input_size=(3, 32, 112, 112)))
    # # Step 1: Initialize model with the best available weights
    # model = mc3_18()
    # print(summary(r2plus1d_18(), input_size=(3, 40, 112, 112)))
    # print(summary(model,input_size=(3, 40, 112, 112)))
    print(summary(MC3_18_simple(num_classes=3), input_size=(3, 40, 112, 112)))
