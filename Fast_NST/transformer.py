import torch

class TransformerNet(torch.nn.Module):
    # following structure described in 
    # https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    # replacing batch norm with instance norm, as the contrast of individual 
    # instances is an important feature in style transfer
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.relu = torch.nn.ReLU()
        
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine = True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine = True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine = True)
        
        self.res = ResidualBlock(128)
        self.res_cnt = 5
        
        self.deconv1 = UpsampleConvLayer(128, 64, 3, 1, 2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine = True)
        self.deconv2 = UpsampleConvLayer(64, 32, 3, 1, 2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine = True)
        self.deconv3 = UpsampleConvLayer(32, 3, 9, 1)

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        for i in range(self.res_cnt):
            y = self.res(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        pad = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(pad)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out


class ResidualBlock(torch.nn.Module):
    # According to paper, in most cases the output image should
    # share structure with the input image. This is why residual
    # connection is introduced, but it doesn't have to be so 
    # complicated as what's done in ResNet.
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine = True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine = True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = residual + out
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample = None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        pad = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(pad)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample != None:
            x_in = torch.nn.functional.interpolate(x_in, mode = 'nearest', scale_factor = self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv(out)
        return out