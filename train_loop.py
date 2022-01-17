import jittor as jt
from loss import StyleGANLoss
from dataloader import get_dataset

from net import Generator, Discriminator
import argparse
import math
import random
from util import ema_update, setup_optim, generate_latent
import os

jt.flags.use_cuda = True
jt.flags.log_silent = True



def main(args):
    import importlib
    config = importlib.import_module(args.config[:-3])
    cfg = config.cfg

    

    gen_net = Generator()
    gen_net_ema = Generator()
    gen_net_ema.eval()
    dis_net = Discriminator()
    style_loss = StyleGANLoss()
    
    
    dis_optim = jt.optim.Adam(dis_net.parameters(), lr=cfg.lr, betas=(0.0, 0.99))
    gen_optim = jt.optim.Adam(gen_net.gsynsis_module.parameters(), lr=cfg.lr, betas=(0.0, 0.99))
    gen_optim.add_param_group({
        'params': gen_net.mapping_module.parameters(),
        'lr': cfg.lr * 0.01,
        'mult': 0.01,
        }
    )

    ema_update(gen_net, gen_net_ema, 0)
    if hasattr(cfg, "load_path"):
        print("loading file: {}".format(cfg.load_path))
        trained_model = jt.load(cfg.load_path)
        gen_net.load_state_dict(trained_model["gen"])
        dis_net.load_state_dict(trained_model["dis"])
        gen_net_ema.load_state_dict(trained_model["gen_ema"])

    


    batch_epoch_schedule = cfg.batch_epoch_schedule

    start_resolution = cfg.start_resolution
    start_idx = int(math.log2(start_resolution) - 2)
    max_resolution = cfg.max_resolution
    end_idx = int(math.log2(max_resolution) - 2)
    batch_epoch_schedule = cfg.batch_epoch_schedule
    data_root_path = cfg.data_root_path


    exp_path = cfg.exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log_file = os.path.join(exp_path, 'train.log')

    f_write = open(log_file, 'w')


    
    gen_loss_val = 0
    disc_loss_val = 0
    grad_loss_val = 0

    resolution = start_resolution
    for cur_idx in range(start_idx, end_idx + 1):
        batch_size, epoch_num = batch_epoch_schedule[cur_idx - 1]
        data_loader = get_dataset(data_root_path, resolution, batch_size)

        epoch_iters = len(data_loader) // batch_size
        # fade_ratio: 0.5
        fade_iters = cfg.fade_ratio * epoch_num * epoch_iters
        

        
        acc_iters = 0
        # import pdb; pdb.set_trace()
        gen_net.eval()
        dis_net.train()
        for ep in range(epoch_num):
            for iters, real_imgs in enumerate(data_loader):
                # mix_ratio 0.9
                # import pdb; pdb.set_trace()
                if cfg.mix_style and random.random() < cfg.mix_ratio:
                    latent = generate_latent(batch_size, mul=4)
                    f1_latent = latent[:2*batch_size]
                    f2_latent = latent[2*batch_size:]
                else:
                    latent = generate_latent(batch_size, mul=2)
                    f1_latent = latent[:batch_size]
                    f2_latent = latent[batch_size:]
                


                acc_iters += 1
                cur_weight = min(acc_iters / float(fade_iters), 1)
                if cur_idx == start_idx or cur_idx == end_idx:
                    cur_weight = 1
         

                # -----------------
                #  Train Discriminator
                # -----------------

                real_imgs.requires_grad = True
                r_out = dis_net(real_imgs, resolution, cur_weight)
                
                
                fake_imgs = gen_net(batch_size, resolution, cur_weight, f1_latent)
                f_out = dis_net(fake_imgs, resolution, cur_weight)

                loss_dis, grad_p = style_loss.dis_loss(real_imgs, r_out, f_out)
                if iters % cfg.print_feq == 0:
                    grad_loss_val = grad_p.item()

                if iters % cfg.print_feq == 0:
                    disc_loss_val = loss_dis.item()

                d_loss = loss_dis + grad_p
                dis_optim.step(d_loss)

                # ---------------------
                #  Train Generator
                # ---------------------

                gen_net.train()
                dis_net.eval()
                fake_imgs = gen_net(batch_size, resolution, cur_weight, f2_latent)
                f_out = dis_net(fake_imgs, resolution, cur_weight)

                g_loss = style_loss.gen_loss(f_out)

                if iters % cfg.print_feq  == 0:
                    gen_loss_val = g_loss.item()

                gen_optim.step(g_loss)
                ema_update(gen_net, gen_net_ema)
                gen_net.eval()
                dis_net.train()

                if iters % cfg.save_img_feq == 0:
                    save_imgs = [[] for _ in range(cfg.save_img_row)]
                    with jt.no_grad():
                        for row in range(cfg.save_img_row):
                            s_latent = jt.randn(cfg.save_img_column, 512)
                            save_imgs[row] = gen_net_ema(cfg.save_img_column, resolution, cur_weight, s_latent).data
                        save_path = os.path.join(exp_path, "samples")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        jt.save_image(jt.concat(save_imgs, 0), "{}/{}_{}.png".format(save_path, resolution, acc_iters), nrow=cfg.save_img_row,
                        normalize=True, range=(-1, 1))
                    pass

                if iters % cfg.print_feq == 0:
                    log_str = "Resolution: {}, Epoch {} // {}, Iters: {} // {}, Gen loss {}, Dis loss val: {}, grad: {}, weight {}".format(
                        resolution, ep, epoch_num, iters, epoch_iters, gen_loss_val, disc_loss_val, grad_loss_val, cur_weight
                    )
                    print(log_str)
                    f_write.write(log_str + "\n")
                    f_write.flush()
                

            # save model per epoch
            model_save_path = os.path.join(exp_path, "models")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            jt.save(
                {
                    'gen': gen_net.state_dict(),
                    'dis': dis_net.state_dict(),
                    'gen_ema': gen_net_ema.state_dict(),
                },
                '{}/resolution{}_ep{}'.format(model_save_path, resolution, ep),
            )


        resolution = resolution * 2
    
    f_write.close()
    print("Training done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jittor StyleGAN Training')
    parser.add_argument('--config', type=str, help='py config file')
    main(parser.parse_args())
