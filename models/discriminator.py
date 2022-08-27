"""
Discriminator Network Definition

"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, module_type, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, dropout_p=0.0, norm=True, activation=True):
        super().__init__()
        if module_type == 'convolution':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        else:
            raise NotImplementedError(f"Module type '{module_type}' is not valid")

        self.lrelu = nn.LeakyReLU(0.2) if activation else None
        self.norm = LayerNorm(out_channels, eps=1e-6,data_format="channels_first") if norm else None

    def forward(self, x):
        x = self.lrelu(x) if self.lrelu else x
        x = self.conv(x)
        x = self.norm(x) if self.norm else x
        return x

class PatchGAN(nn.Module):
    def __init__(self, in_channels=260, out_channels=1, bias=False, norm=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        
        self.discriminator_blocks = nn.ModuleList([
            DiscriminatorBlock('convolution', 256, 64, bias=bias, norm=False, activation=False),
            DiscriminatorBlock('convolution', 64, 128, bias=bias, norm=True, activation=True),
            DiscriminatorBlock('convolution', 128, 256, bias=bias, norm=True, activation=True),
            DiscriminatorBlock('convolution', 256, 512, bias=bias, norm=True, activation=True),
            DiscriminatorBlock('convolution', 512, out_channels, bias=bias, norm=False, stride=1)
        ])
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, mask, backbone):
        try:
            output = torch.cat([img, backbone[0], mask], 1)
        except:
            print("size mismatch!!")
            print("OMKAR $$$$$$$$")

            return None
        
        output = self.conv1(output)
        output = self.pool(output)
        output = output + backbone[1]
        
        output = self.conv2(output)
        output = self.pool(output)
        output = output + backbone[2]
        
        
        for block in self.discriminator_blocks:
            output = block(output)
        
        output = self.sigmoid(output)
        return output
