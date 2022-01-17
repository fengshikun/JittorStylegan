import argparse
import math

from net import Generator
import os

import jittor as jt
jt.flags.use_cuda = True
jt.flags.log_silent = True




def generate_latent(batch_size, dim=512, mul=2):
    # latent_b = batch_size * 2 if mix else batch_size
    latent = jt.randn(batch_size * mul, dim)
    return latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128, help='image size')
    parser.add_argument('--row', type=int, default=6, help='rows')
    parser.add_argument('--col', type=int, default=6, help='cols')
    parser.add_argument('--trunc', dest='trunc', action='store_true')
    parser.add_argument('--model', type=str, help='path to checkpoint file')
    
    args = parser.parse_args()

    gen_net = Generator(args.trunc)
    ckpt = jt.load(args.model)
    gen_net.load_state_dict(ckpt["gen"])
    
    # gen_net.load_state_dict(ckpt["gen"])
    gen_net.eval()
    gen_net.generate_mean()


    step = int(math.log(args.size, 2)) - 2
    
    if not os.path.exists("mix_style"):
        os.makedirs("mix_style")

    # img = sample(generator, step, mean_style, args.n_row * args.n_col)
    # jt.save_image(img, 'style_mixing/sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
    for j in range(20):
        batch_size = args.row
        latent = generate_latent(batch_size, mul=2)
        f1_latent = latent[:batch_size]
        f2_latent = latent[batch_size:]

        source_img = gen_net(batch_size, args.size, 1, f1_latent)
        target_img = gen_net(batch_size, args.size, 1, f2_latent)
        
        images = [jt.ones((1, 3, args.size, args.size))]
        
        images.append(source_img)

        for i in range(args.col):
            mix_latent = jt.randn(batch_size * 2, 512)
            mix_latent[:batch_size] = f1_latent
            mix_latent[batch_size:] = f2_latent[i].unsqueeze(0).repeat(batch_size, 1)
            # import pdb; pdb.set_trace()
            image = gen_net(batch_size, args.size, 1, mix_latent, mix_specify=1).data
            
            images.append(target_img[i].unsqueeze(0))
            images.append(image)

        images = jt.concat(images, 0)
        
        jt.save_image(
            images, f'mix_style/res_{j}.png', nrow=args.col + 1, normalize=True, range=(-1, 1)
        )