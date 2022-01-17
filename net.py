from customlayers import *
from block import *
import jittor as jt
import numpy as np
import random
from math import sqrt


class Discriminator(jt.Module):
    def __init__(self, 
    body_length=8, in_channel=16, output_channel=512, kernel_size=3, padding=1):
        self.body_length = body_length
        self.backbone = nn.ModuleList()
        self.rgb_head = nn.ModuleList()
        
        start = 16
        for i in range(body_length):
            if start == output_channel:
                out_c = output_channel
            else:
                out_c = start * 2
            if start < output_channel / 2:
                scale_conv = True
            else:
                scale_conv = False

            self.backbone.append(DBlock(start, out_c, kernel_size, padding, final=False, scale_conv=scale_conv))
            self.rgb_head.append(FromRgb(start))
            
            if start < output_channel:
                start = start * 2
        # head
        assert start == output_channel
        # self.head = nn.Sequential(MinibatchStdDev(), DBlock(output_channel + 1, output_channel, kernel_size, padding, final=True))
        self.head = DBlock(output_channel + 1, output_channel, kernel_size, padding, final=True)
        
        self.n_layer = body_length + 1

        self.rgb_head.append(FromRgb(output_channel))
        self.predict = EqualizedLinear(output_channel, 1)
    
    def execute(self, img, resolution, cur_weight):
        # body_length: 8
        start_idx = int(self.body_length - (math.log2(resolution) - 2))
        # 8-->7
        for idx in range(start_idx, self.body_length + 1):
            if idx == start_idx:
                x = self.rgb_head[idx](img)
                x = self.backbone[idx](x)
                if cur_weight < 1:
                    previous = nn.avg_pool2d(img, 2)
                    previous = self.rgb_head[idx + 1](previous)
                    x = (1 - cur_weight) * previous + cur_weight * x
            elif idx == self.body_length:
                out_std = np.std(x.data, axis=0)
                mean_std = jt.array(out_std.mean())
                mean_std = mean_std.expand((x.size(0), 1, 4, 4))
                x = jt.concat([x, mean_std], 1)
                x = self.head(x)
            else:
                x = self.backbone[idx](x)

        x = x.squeeze(2).squeeze(2) # todo change to reshape
        x = self.predict(x)
        return x



class GSynthesis(jt.Module):
    def __init__(self, body_length=9, in_channel=512, output_channel=16, kernel_size=3, padding=1):
        self.learnable_input = ConstantModule(in_channel)
        down_degree = int(math.log2(in_channel // output_channel))
        self.body_module = nn.ModuleList()
        self.to_rgb_module = nn.ModuleList()
        for i in range(0, body_length - down_degree):
            if i == 0:
                self.body_module.append(GBlock(in_channel, in_channel, kernel_size, padding, first_block=True))
            else:
                self.body_module.append(GBlock(in_channel, in_channel, kernel_size, padding, scale_conv=False))
            self.to_rgb_module.append(EqualizedConv2d(in_channel, 3, 1))

        start_channel = in_channel
        for i in range(body_length - down_degree, body_length):
            if i == body_length - down_degree:
                scale_conv = False
            else:
                scale_conv = True
            self.body_module.append(GBlock(start_channel, start_channel//2, kernel_size, padding, scale_conv=scale_conv))
            start_channel = start_channel // 2
            self.to_rgb_module.append(EqualizedConv2d(start_channel, 3, 1))
        assert(start_channel == output_channel)


       

    def generate_noise(self, resolution, batch_size, min_size=4):
        noise = []
        start_rl = 4
        while start_rl <= resolution:
            noise.append(jt.randn(batch_size, 1, start_rl, start_rl))
            start_rl = start_rl * 2
        return noise

    def execute(self, w_latent, batch_size, resolution, cur_weight, mix_specify=-1):
        noise = self.generate_noise(resolution, batch_size)
        learn_input = self.learnable_input(batch_size)
        
        depth = int((math.log2(resolution) - 2)) + 1 # 4 * 4: 1 ;8 * 8: 2;
        
        w_latent_array = []
        if isinstance(w_latent, list):
            w_latent_array = w_latent
            mix_idx = random.sample([i for i in range(depth - 1)], 1)[0]
        else:
            w_latent_array.append(w_latent)

        w_input = w_latent_array[0]

        for d in range(depth):
            # mix style
            if len(w_latent_array) > 1:
                if mix_specify > 0:
                    if d > mix_specify:
                        w_input = w_latent_array[1]
                elif d > mix_idx:
                    w_input = w_latent_array[1]
            
            if cur_weight < 1 and d > 0:
                learn_prev = learn_input
            learn_input = self.body_module[d](learn_input, w_input, noise[d])
            if d == depth - 1:
                g_img = self.to_rgb_module[d](learn_input)
                if cur_weight < 1:
                    previous = self.to_rgb_module[d - 1](learn_prev)
                    previous = nn.interpolate(previous, scale_factor=2, mode='nearest')
                    g_img = (1 - cur_weight) * previous + cur_weight * g_img
                break

        return g_img

    

class Generator(nn.Module):
    def __init__(self, trunc=False):
        super(Generator, self).__init__()
        self.mapping_module = DenseMapping()
        self.trunc = trunc
        if self.trunc:
            # self._generate_mean() # call after load
            self._pi = 0.7
            # self.trunc_layer = Truncation(jt.zeros(512))
        self.gsynsis_module = GSynthesis()



    def generate_mean(self):
        iters = 100
        batch_size = 1000
        style_mean = jt.zeros(512)
        for i in range(iters):
            batch_style = self.mapping_module(jt.randn((batch_size, 512))).mean(0, keepdims=True)
            style_mean += batch_style

        # import pdb; pdb.set_trace()
        self._style_mean = style_mean / iters
    
    def _trunc_style(self, style):
        return self._style_mean + self._pi * (style - self._style_mean)
        

    def execute(self, batch_size, resolution, cur_weight, z_latent, mix_specify=-1):
        # import pdb; pdb.set_trace()
        cur_batch_size = z_latent.shape[0]
        if cur_batch_size == 2 * batch_size:
        # if isinstance(z_latent, list):
            # z_latent1 = self.generate_latent(batch_size)
            # z_latent2 = self.generate_latent(batch_size)

            z_latent1, z_latent2 = z_latent[:batch_size], z_latent[batch_size:] 
            

            w_latent1 = self.mapping_module(z_latent1)
            w_latent2 = self.mapping_module(z_latent2)
            # w_latent1 = self.mapping_module(z_latent[0])
            # w_latent2 = self.mapping_module(z_latent[1])
            if self.trunc:
                w_latent1 = self._trunc_style(w_latent1)
                w_latent2 = self._trunc_style(w_latent2)

            w_latent = [w_latent1, w_latent2]
        else:
            # z_latent = self.generate_latent(batch_size, mix=self.mix_style)
            
            w_latent = self.mapping_module(z_latent)
            if self.trunc:
                w_latent = self._trunc_style(w_latent)
       
        g_img = self.gsynsis_module(w_latent, batch_size, resolution, cur_weight, mix_specify=mix_specify)
        return g_img


