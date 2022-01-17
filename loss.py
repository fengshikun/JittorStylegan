
import jittor as jt
import jittor.nn as nn



class StyleGANLoss(nn.Module):
    def __init__(self, wgan_lambda = 5):
        super(StyleGANLoss, self).__init__()
        self.wgan_lambda = 5

        
    def gen_loss(self, f_out):
        # f_out = nn.Sigmoid()(-f_out) * 5
        return jt.nn.softplus(-f_out).mean()

    def dis_loss(self, r_img, r_out, f_out):
        grads = jt.grad(r_out.sum(), r_img)
        r1_p = jt.reshape(grads, (grads.shape[0], -1)).norm(2, dim=1).sqr().mean()

        return jt.nn.softplus(-r_out).mean() + jt.nn.softplus(f_out).mean(), self.wgan_lambda * r1_p
        # return d_loss

