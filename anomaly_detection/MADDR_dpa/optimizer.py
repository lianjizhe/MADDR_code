from torch import nn
from torch.optim import Adam
import torch.optim as optim
from torch.optim import lr_scheduler

class Optimizer(nn.Module):
    def __init__(self, enc_params, dec_params, ddis_params,
                 image_adv_loss, image_rec_loss,
                 image_adv_weight, image_rec_weight, adam_kwargs):
        super(Optimizer, self).__init__()

        self.adam_kwargs = adam_kwargs
        self.image_adv_loss = image_adv_loss
        self.image_rec_loss = image_rec_loss

        self.image_adv_weight = image_adv_weight
        self.image_rec_weight = image_rec_weight

        self.enc_dec_opt, self.ddis_opt = None, None

        self.set_new_params(enc_params, dec_params, ddis_params)

    def set_new_params(self, enc_params, dec_params, ddis_params, image_rec_loss=None):

        def preprocess(params):
            params = [p for p in params if p.requires_grad]
            if len(params) == 0:
                params = nn.Conv2d(1, 1, 1).parameters()
            return params

        ddis_params = preprocess(ddis_params)
        enc_dec_params = preprocess(enc_params) + preprocess(dec_params)

        self.ddis_opt = Adam(ddis_params, **self.adam_kwargs)
        self.enc_dec_opt = Adam(enc_dec_params, **self.adam_kwargs)
        self.scheduler = lr_scheduler.StepLR(self.enc_dec_opt , step_size=10000, gamma=0.8)

        if image_rec_loss is not None:
            self.image_rec_loss = image_rec_loss

    def compute_ddis_loss(self, ddis, real, fake, update_parameters=True):
        if self.image_adv_weight == 0:
            return {}

        loss, loss_info = self.image_adv_loss.dis_loss(ddis, real.detach(), fake.detach())

        if update_parameters:
            self.ddis_opt.zero_grad()
            loss.backward()
            self.ddis_opt.step()

        return loss_info

    def compute_adv_loss(self, ddis, real, fake):
        if self.image_adv_weight == 0:
            return None

        loss, loss_info = self.image_adv_loss.gen_loss(ddis, real, fake)
        return self.image_adv_weight * loss

    def compute_rec_loss(self, real, fake_z, dec, label, zs, zc_logit):
        if self.image_rec_weight == 0:
            return None
        loss = self.image_rec_loss(real, fake_z, dec, label, zs, zc_logit)
        return self.image_rec_weight * loss

    def compute_enc_dec_loss(self, ddis, real_x, fake_z, dec, label_x, zs, zc_logit, update_parameters=True):

        # adv_loss = self.compute_adv_loss(ddis, real=None, fake=rec_x)
        rec_loss = self.compute_rec_loss(real=real_x, fake_z=fake_z, dec=dec, label=label_x, zs=zs, zc_logit=zc_logit)

        losses = {
            # 'image_adv_loss': adv_loss,
            'image_rec_loss': rec_loss
        }

        total_loss = 0
        loss_info = {}

        for name, loss in losses.items():
            if loss is not None:
                total_loss = total_loss + loss
                loss_info[name] = loss.item()

        if update_parameters:
            if total_loss.item() > 1e6:
                print(total_loss)
                raise ValueError("Too large value of loss function (>10^6)!")

            self.enc_dec_opt.zero_grad()
            total_loss.backward()
            self.enc_dec_opt.step()
            self.scheduler.step()
        return loss_info

    def compute_enc_dec_loss_(self, ddis, real_x, fake_z, dec, zs, zc_logit, targets_a, targets_b, lam, update_parameters=True):

        # adv_loss = self.compute_adv_loss(ddis, real=None, fake=rec_x)
        rec_a_loss = self.compute_rec_loss(real=real_x, fake_z=fake_z, dec=dec, label=targets_a, zs=zs, zc_logit=zc_logit)
        rec_b_loss = self.compute_rec_loss(real=real_x, fake_z=fake_z, dec=dec, label=targets_b, zs=zs, zc_logit=zc_logit)

        losses = {
            'image_rec_a_loss': rec_a_loss,
            'image_rec_b_loss': rec_b_loss
        }

        total_loss = 0
        loss_info = {}

        for name, loss in losses.items():
            if loss is not None:
                # total_loss = total_loss + loss
                loss_info[name] = loss.item()

        total_loss = lam * rec_a_loss + (1 - lam) * rec_b_loss
        if update_parameters:
            self.enc_dec_opt.zero_grad()
            total_loss.backward()
            self.enc_dec_opt.step()
            # self.scheduler.step()
        return loss_info
