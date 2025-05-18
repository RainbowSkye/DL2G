import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')

# 模型
class DDPM(BaseModel):
    def __init__(self, opt, m_items):
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_G(opt, m_items))
        self.schedule_phase = None

        self.set_loss()
        self.set_loss_mask()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    # v.requires_grad = False           # refineCN
                    # if k.find('EFA') >= 0 or k.find('RefineModule') >= 0 or k.find('DualFusion') >= 0:
                    #     v.requires_grad = True
                    v.requires_grad = True              # refineDM
                    if k.find('EFA') >= 0 or k.find('RefineModule') >= 0 or k.find('DualFusion') >= 0:
                        v.requires_grad = False
                optim_params = list(self.netG.parameters())
                # self.optG = torch.optim.Adam(         # refineCN
                #     optim_params, lr=0.001)
                # print("init>>>LR: {}".format(str(self.optG.param_groups[0]['lr'])))
                # scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optG, gamma=0.95)
                # self.scheduler = scheduler

                self.optG = torch.optim.Adam(           # refineDM
                    optim_params, lr=opt['train']["optimizer"]["lr"])
                self.scheduler = None
            else:
                optim_params = list(self.netG.parameters())
                self.optG = torch.optim.Adam(
                    optim_params, lr=opt['train']["optimizer"]["lr"])
                self.scheduler = None
            self.log_dict = OrderedDict()
        self.load_network()
        # self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(x=self.data)
        l_pix = l_pix.sum()
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def optimize_optimizer(self, current_epoch):
        print("epoch: {}, LR: {}".format(str(current_epoch), str(self.optG.param_groups[0]['lr'])))
        self.scheduler.step()
        print("scheduled>>>epoch: {}, LR: {}".format(str(current_epoch), str(self.optG.param_groups[0]['lr'])))

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.Result = []
                self.Result = self.netG.module.super_resolution(
                    self.data, continous)
            else:
                self.Result = []
                self.Result = self.netG.super_resolution(
                    self.data, continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.Input = self.netG.module.sample(batch_size, continous)
            else:
                self.Input = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_loss_mask(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss_mask(self.device)
        else:
            self.netG.set_loss_mask(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.Result[0].detach().float().cpu()
        else:
            out_dict['Result'] = self.Result[0].detach().float().cpu()
            # out_dict['Zt'] = self.Result[2].detach().float().cpu()
            # out_dict['Mask'] = self.Result[2].detach().float().cpu()
            # out_dict['Spec'] = self.Result[2].detach().float().cpu()
            out_dict['Input'] = self.data['Input'].detach().float().cpu()
            out_dict['GT'] = self.data['GT'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        items_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_items.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        opt_state['scheduler'] = self.scheduler.state_dict()
        torch.save(opt_state, opt_path)

        torch.save(network.m_items, items_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            items_path = '{}_items.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            self.netG.m_items = torch.load(items_path)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                # if self.opt['model']['finetune_norm']:            # refineCN
                #     print("load after>>>LR: {}".format(str(self.optG.param_groups[0]['lr'])))
                #     self.optG.param_groups[0]['lr'] = 0.0001
                #     print("reset>>>LR: {}".format(str(self.optG.param_groups[0]['lr'])))
                #     self.scheduler.load_state_dict(opt['scheduler'])
