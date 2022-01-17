import jittor as jt
import jittor.nn as nn

from customlayers import *


class Truncation(nn.Module):
    def __init__(self, mean_w, pi=0.7, mom=0.99):
        super(Truncation, self).__init__()
        self.pi = pi
        self.mom = mom
        self._mean_w = mean_w

    def update(self, mean_w):
        self._mean_w = (self.mom * self._mean_w + (1. - self.mom) * mean_w)

    def execute(self, x):
        self.update(x.mean(dim=0))
        trunc_w = self._mean_w + self.pi * (x - self._mean_w)
        return trunc_w

class DenseMapping(nn.Module):
    def __init__(self, latent_size=512, fc_len=8):
        super(DenseMapping, self).__init__()
        self.norm_layer = PixelNorm()
        self.fc_layer = nn.Sequential()
        for i in range(fc_len):
            self.fc_layer.append(EqualizedLinear(latent_size, latent_size))
            self.fc_layer.append(nn.LeakyReLU(scale=0.2))

    
    def execute(self, z):
        z = self.norm_layer(z)
        z = self.fc_layer(z)
        return z


class FromRgb(nn.Module):
    def __init__(self, out_feature, kernel_size=1, scale=0.2):
        super(FromRgb, self).__init__()
        self.module_list = nn.Sequential(
            EqualizedConv2d(3, out_feature, kernel_size),
            nn.LeakyReLU(scale)
        )
    
    def execute(self, x):
        x = self.module_list(x)
        return x
    


class DBlock(nn.Module):
    def __init__(self, features_in, features_out, kernel_size, padding, 
        final=False, 
        scale_conv=False, 
        scale=0.2,
        final_kernel_size=4
    ):
        super(DBlock, self).__init__()
        self.module_list = nn.Sequential(
            EqualizedConv2d(features_in, features_out, kernel_size, padding=padding),
            nn.LeakyReLU(scale)
        )
        if final:
            self.module_list.append(EqualizedConv2d(features_out, features_out, final_kernel_size, padding=0))
        else:
            self.module_list.append(Blur2d(features_out))
            if scale_conv:
                self.module_list.append(ScaleConv(True, features_out, features_out, kernel_size, padding=padding))
            else:
                self.module_list.append(EqualizedConv2d(features_out, features_out, kernel_size, padding=padding))
                self.module_list.append(nn.AvgPool2d(2))
        self.module_list.append(nn.LeakyReLU(scale))
        
    def execute(self, x):
        x = self.module_list(x)
        return x

class GBlock(nn.Module):
    def __init__(self, features_in, features_out, kernel_size, padding, 
        style_dim=512,
        first_block = False,
        scale_conv = False
    ):
        super(GBlock, self).__init__()
        self.conv1 = nn.Sequential()
        if not first_block:
            if scale_conv:
                self.conv1.append(ScaleConv(False, features_in, features_out, kernel_size, stride=2, padding=padding))
            else:
                self.conv1.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.conv1.append(EqualizedConv2d(features_in, features_out, kernel_size, padding=padding))
            self.conv1.append(Blur2d(features_out))
        
        self.conv2 = EqualizedConv2d(features_out, features_out, kernel_size, padding=padding)
        self.nm1 = NoiseModule(features_out)
        self.as1 = AdaINStye(features_out, style_dim=style_dim)
        self.nm2 = NoiseModule(features_out)
        self.as2 = AdaINStye(features_out, style_dim=style_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.act2 = nn.LeakyReLU(0.2)


    
    def execute(self, x, style_emb, noise_emb):
        x = self.conv1(x)
        x = self.nm1(x, noise_emb)
        x = self.act1(x)
        x = self.as1(x, style_emb)

        x = self.conv2(x)
        x = self.nm2(x, noise_emb)
        x = self.act2(x)
        x = self.as2(x, style_emb)

        return x

