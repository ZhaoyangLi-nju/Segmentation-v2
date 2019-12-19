import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from datetime import datetime

class BaseModel(nn.Module):

    def name(self):
        pass

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.model = None
        self.device = torch.device('cuda' if self.cfg.GPU_IDS else 'cpu')
        self.save_dir = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.MODEL)
        # , str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())
        if not os.path.exists(self.save_dir):
            # shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

    # schedule for modifying learning rate
    def set_schedulers(self, cfg):

        self.schedulers = [self._get_scheduler(optimizer, cfg, cfg.LR_POLICY) for optimizer in self.optimizers]

    def _get_scheduler(self, optimizer, cfg, lr_policy):
        if lr_policy == 'lambda':
            print('use lambda lr')
            decay_start = cfg.NITER
            decay_iters = cfg.NITER_DECAY
            total_iters = cfg.NITER_TOTAL

            # assert NITER_TOTAL == decay_start + decay_iters

            def lambda_rule(iters):
                # lr_l = (1 - float(iters) / total_iters) ** 0.8

                lr_l = 1 - max(0, iters - decay_start) / float(decay_iters)

                # warmup
                if cfg.PRETRAINED != 'imagenet' and cfg.PRETRAINED != 'place' and iters < 500:
                    lr_scale = min(1., float(iters + 1) / 500.)
                    lr_l = lr_l * lr_scale

                # lr_l = pow(2, - (iters // decay_start))

                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            print('use step lr')
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.LR_DECAY_ITERS, gamma=0.1)
        elif lr_policy == 'plateau':
            print('use plateau lr')
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                                       threshold=0.0001, factor=0.5, patience=2, eps=1e-7)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
        return scheduler

    def set_data_loader(self, train_loader=None, val_loader=None):

        if train_loader is not None:
            self.train_loader = train_loader
            self.train_image_num = self.train_loader.dataset.__len__()

            print('train_num:',self.train_image_num)
        if val_loader is not None:
            self.val_loader = val_loader
            self.val_image_num = self.val_loader.dataset.__len__()
            print('val_num:', self.val_image_num)

    def load_checkpoint(self, load_path, keep_fc=False):
        state_checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage.cuda())
        print('loading {0} ...'.format(load_path))
        if os.path.isfile(load_path):
            for name in self.model_names:

                net = getattr(self, name)
                state_dict = net.state_dict()

                if 'd_distribute' in name or 'discriminator' in name:
                    continue

                for k, v in state_checkpoint[name].items():
                    k = str.replace(k, 'module.', '')
                    if 'd_cross' in k:
                        continue
                    if k in state_dict.keys():
                        if not keep_fc:
                            if 'fc' in k or 'evaluator' in k:
                                continue
                        state_dict[k] = v

                # state_dict.update(state_checkpoint)
                net.load_state_dict(state_dict)
        else:
            raise ValueError('No checkpoint found at {0}'.format(load_path))


    def _load_checkpoint(self, net, load_path, key='net', keep_fc=False):
        state_checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage.cuda())
        print('loading {0} ...'.format(load_path))
        if os.path.isfile(load_path):
            state_dict = net.state_dict()

            if key is not None:
                state_checkpoint = state_checkpoint[key]
            for k, v in state_checkpoint.items():
                k = str.replace(k, 'module.', '')
                if k in state_dict.keys():
                    if not keep_fc:
                        if 'fc' in k or 'evaluator' in k:
                            continue
                    state_dict[k] = v

            net.load_state_dict(state_dict)
        else:
            raise ValueError('No checkpoint found at {0}'.format(load_path))

    def save_checkpoint(self, filename=None):
        state = dict()
        if filename is None:
            filename = '{0}.pth'.format(self.cfg.LOG_NAME)
            # filename = '{0}_{1}_{2}_{3}.pth'.format(self.cfg.MODEL, self.cfg.TASK, str(self.cfg.NITER_TOTAL),
            #                                             datetime.now().strftime('%b%d_%H-%M'))
        for name in self.model_names:

            net = getattr(self, name)
            net_state_dict = net.state_dict()
            save_state_dict = {}
            for k, v in net_state_dict.items():
                if 'content_model' in k or 'sample_model' in k:
                    continue
                save_state_dict[k] = v
            state[name] = save_state_dict

        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            print('save file {0}'.format(filename))
            torch.save(state, filepath)

    def update_learning_rate(self, val=None, step=None):
        for scheduler in self.schedulers:
            if val is not None:
                scheduler.step(val)
            else:
                scheduler.step(step)

    def print_lr(self):
        for optimizer in self.optimizers:
            # print('default lr', optimizer.defaults['lr'])
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                print('/////////learning rate = %.7f' % lr)

    def set_log_data(self, cfg):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for name, param in net.named_parameters():
                    if 'content_model' in name:
                        param.requires_grad = False
                        continue
                    param.requires_grad = requires_grad


