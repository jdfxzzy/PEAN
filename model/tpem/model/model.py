from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.tpem.model.networks as networks
from model.tpem.model.base_model import BaseModel


class TPEM(BaseModel):
    def __init__(self, opt):
        super(TPEM, self).__init__(opt)

        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            for p in self.netG.parameters():
                p.requires_grad = True
        else:
            self.netG.eval()
            for p in self.netG.parameters():
                p.requires_grad = False

        self.load_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def process(self):
        loss_pix, loss_ctc, x_recon = self.netG(self.data)
        b, c, l = self.data['HR'].shape
        loss_pix = loss_pix.sum() / int(b*c*l)

        return loss_pix + loss_ctc, x_recon

    def test(self, sample_method="DDIM"):
        self.netG.eval()
        for p in self.netG.parameters():
            p.requires_grad = False
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(self.data['SR'], sample_method)
            else:
                self.SR = self.netG.super_resolution(self.data['SR'], sample_method)
        self.netG.train()
        for p in self.netG.parameters():
            p.requires_grad = True

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)


    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict


    def save_network(self, epoch, iter_step, recognizer):
        gen_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_{}_gen.pth'.format(iter_step, epoch, recognizer))
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)


    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            print("loading TPEM from {}".format(load_path))
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(load_path))