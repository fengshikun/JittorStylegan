import jittor as jt
from jittor import init

def linear_init(linear):
    init.gauss_(linear.weight, 0, 1)
    jt.init.constant_(linear.bias, 0)


def conv_init(conv):
    init.gauss_(conv.weight, 0, 1)
    if conv.bias is not None:
        jt.init.constant_(conv.bias, 0)

def ema_update(source, target, mom=0.99):
    source_params_dict = dict(source.named_parameters())
    for p_name, s_param in target.named_parameters():
        s_param.update(s_param * mom + (1 - mom) * source_params_dict[p_name].detach())

def setup_optim(Gen, Disc, lr):
    gen_optim = jt.optim.Adam(Gen.gsynsis_module.parameters(), lr=lr, betas=(0.0, 0.99))
    gen_optim.add_param_group({"params": Gen.mapping_module.parameters(), lr:lr * 0.01, "mult":0.01})
    dis_optim = jt.optim.Adam(Disc.parameters(), lr=lr, betas=(0.0, 0.99))
    return gen_optim, dis_optim

def generate_latent(batch_size, dim=512, mul=2):
    # latent_b = batch_size * 2 if mix else batch_size
    latent = jt.randn(batch_size * mul, dim)
    return latent