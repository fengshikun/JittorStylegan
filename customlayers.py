import jittor as jt
import jittor.nn as nn
from util import conv_init, linear_init
import math
import random

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
        pass
    def execute(self, x, eps=1e-12):
        batch_size, _, height, width = x.shape
        y = x - x.mean(dim=0, keepdims=True)
        y = jt.sqrt(y.pow(2.).mean(dim=0, keepdims=False) + eps)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, height, width)
        y = jt.concat([x, y], 1)
        return y

class Blur2d(nn.Module):
    def __init__(self, input_channel):
        super(Blur2d, self).__init__()
        kernel = jt.array([1, 2, 1], dtype=jt.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel.reshape(1, 1, 3, 3)
        kernel = kernel / kernel.sum()
        self._kernel = kernel.repeat(input_channel, 1, 1, 1)


    def execute(self, x):
        x = nn.conv2d(x, self._kernel, padding=int((self._kernel.shape[2] - 1) / 2), groups=x.shape[1])
        return x



class EqualizedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EqualizedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(*args, **kwargs)
        conv_init(self.conv2d)

        self.conv2d.t_weight = self.conv2d.weight
        self.f_in = self.conv2d.t_weight[0].numel()
        # inject
        self.conv2d.register_pre_forward_hook(self.equalize)
    
    def equalize(self, conv, input):
        conv.weight = math.sqrt(2 / self.f_in) * conv.t_weight
    
    def execute(self, x):
        x = self.conv2d(x)
        return x
    
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(EqualizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        linear_init(self.linear)
        
        self.linear.t_weight = self.linear.weight
        self.f_in = self.linear.t_weight[0].numel()
        self.linear.register_pre_forward_hook(self.equalize)
    
    def equalize(self, linear, input):
        linear.weight = math.sqrt(2 / self.f_in) * linear.t_weight
    
    def execute(self, x):
        x = self.linear(x)
        return x





# stride = 2
class ScaleConv(nn.Module):
    def __init__(self, down, in_features, out_features, kernel_size, stride=2, padding=0):
        super(ScaleConv, self).__init__()
        self.down = down

        self.weight = jt.randn(in_features, out_features, kernel_size, kernel_size)
        self.bias = jt.zeros(out_features)
        self.f_in = math.sqrt(2 / (in_features * kernel_size * kernel_size))
        self.padding = padding
        
    
    def execute(self, x):

        weight = nn.pad(self.weight * self.f_in, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) * 0.25



        if self.down:
            x= nn.conv2d(x, weight, self.bias, stride=2, padding=self.padding)
        else:
            x = nn.conv_transpose2d(x, weight, self.bias, stride=2, padding=self.padding)
        return x

# refer to style-based-gan-pytorch

class ConstantModule(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstantModule, self).__init__()
        self.input = jt.randn(1, channel, size, size)

    def execute(self, batch_size):
        out = self.input.repeat(batch_size, 1, 1, 1)

        return out

class NoiseModule(nn.Module):
    def __init__(self, channel):
        super(NoiseModule, self).__init__()
        self.weight = jt.zeros((1, channel, 1, 1))
        self.f_in = channel

    def execute(self, image, noise):
        return image + self.weight * jt.sqrt(2 / self.f_in) * noise
        #  * math.sqrt(2 / self.f_in)

class AdaINStye(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(AdaINStye, self).__init__()
        self.norm = jt.nn.InstanceNorm2d(in_channel, affine=False)
        self.style = EqualizedLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def execute(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        pass

    def execute(self, input):
        return input / jt.sqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)