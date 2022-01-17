from easydict import EasyDict as edict

cfg = edict()

cfg.max_resolution = 128
cfg.start_resolution = 8
cfg.lr = 0.001
cfg.mix_style = True
cfg.mix_ratio = 0.9
# cfg.trunc = True
# 8 x 8; 16 x 16; 32 x 32; 64 x 64; 128 x 128
cfg.batch_epoch_schedule = [[256, 6],[128, 6], [64, 6], [32, 6], [16, 32]]
cfg.fade_ratio = 0.5
cfg.n_critic = 1


# cfg.load_path = "/home/fengshikun/JittorGAN/FFHQ/models/resolution128_ep16"
cfg.data_root_path = "/sharefs/sharefs-skfeng/FFHQ"
# cfg.data_root_path = "/data2/skfeng/FFHQ_data"
cfg.exp_path = "FFHQ_new_1227"

cfg.save_img_row = 10
cfg.save_img_column = 6

cfg.print_feq = 10
cfg.save_img_feq = 100



