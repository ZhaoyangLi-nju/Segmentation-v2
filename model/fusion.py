import torch
import torchvision

from util.average_meter import AverageMeter
from .trans2_model import Trans2Net
from . import networks
import apex
import torch.nn.functional as F
import util.utils as util
import torch.distributed as dist
import numpy as np
import torch.nn as nn
from apex.parallel import DistributedDataParallel as DDP
import copy
import os
import random

class Fusion(Trans2Net):

    def __init__(self, cfg, writer=None):
        super(Fusion, self).__init__(cfg, writer)

    def _define_networks(self):
        cfg_tmp = copy.deepcopy(self.cfg)
        cfg_tmp.MODEL = 'trecg_maxpool'
        cfg_tmp.RESUME = False
        cfg_tmp.USE_FAKE_DATA = False
        cfg_tmp.NO_TRANS = True
        self.net_rgb = networks.define_netowrks(cfg_tmp, device=self.device)
        self.net_depth = networks.define_netowrks(cfg_tmp, device=self.device)

        self.model_names = ['net']

        # load parameters
        checkpoint_path_A = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.RESUME_PATH_A)
        checkpoint_path_B = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.RESUME_PATH_B)
        self._load_checkpoint(self.net_rgb, checkpoint_path_A, keep_fc=True)
        self._load_checkpoint(self.net_depth, checkpoint_path_B, keep_fc=True)

        # self.net = networks.Fusion(self.cfg, self.net_rgb, self.net_depth)
        self.net = networks.Fusion(self.cfg, self.net_rgb, self.net_depth)
        networks.print_network(self.net)

        if self.cfg.USE_FAKE_DATA:
            print('Use fake data: sample model 1 is {0}'.format(self.cfg.SAMPLE_PATH_A))
            print('Use fake data: sample model 2 is {0}'.format(self.cfg.SAMPLE_PATH_B))
            checkpoint_path_A = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.SAMPLE_PATH_A)
            # checkpoint_path_B = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.SAMPLE_PATH_B)
            print('fake ratio:', self.cfg.FAKE_DATA_RATE)
            cfg_tmp.USE_FAKE_DATA = False
            cfg_tmp.NO_TRANS = False
            cfg_tmp.MODEL = 'trecg_compl'
            self.sample_model_AtoB = networks.define_netowrks(cfg_tmp, device=self.device)
            # self.sample_model_BtoA = networks.define_netowrks(cfg_tmp, device=self.device)
            self._load_checkpoint(self.sample_model_AtoB, checkpoint_path_A, keep_fc=False)
            # self._load_checkpoint(self.sample_model_BtoA, checkpoint_path_B, keep_fc=False)
            self.sample_model_AtoB.eval()
            # self.sample_model_BtoA.eval()
            self.sample_model_AtoB = nn.DataParallel(self.sample_model_AtoB).to(self.device)
            # self.sample_model_BtoA = nn.DataParallel(self.sample_model_BtoA).to(self.device)

    # def _load_checkpoint(self, net, load_path, keep_fc=False):
    #     state_checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage.cuda())
    #     print('loading {0} ...'.format(load_path))
    #     if os.path.isfile(load_path):
    #
    #         state_dict = net.state_dict()
    #
    #         for k, v in state_checkpoint['net'].items():
    #             k = str.replace(k, 'module.', '')
    #             if k in state_dict.keys():
    #                 if not keep_fc:
    #                     if 'fc' in k or 'evaluator' in k:
    #                         continue
    #                 state_dict[k] = v
    #
    #         # state_dict.update(state_checkpoint)
    #         net.load_state_dict(state_dict)
    #     else:
    #         raise ValueError('No checkpoint found at {0}'.format(load_path))

    def set_input(self, data, d_type='pair'):
        input_A = data['image']
        self.input_rgb = input_A.to(self.device)

        if 'depth' in data:
            input_B = data['depth']
            self.input_depth = input_B.to(self.device)
        else:
            self.input_depth = copy.deepcopy(self.input_rgb)

        self.batch_size = input_A.size(0)

        if 'label' in data.keys():
            self._label = data['label']
            self.label = torch.LongTensor(self._label).to(self.device)

    def _forward(self, cal_loss=True):

        if self.cfg.USE_FAKE_DATA:
            with torch.no_grad():
                result_sample_AtoB = self.sample_model_AtoB(source=self.input_rgb, target=None, label=None, phase=self.phase,
                               cal_loss=False)
                self.fake_depth = result_sample_AtoB['gen_img']
                # [self.fake_rgb] = self.sample_model_BtoA(source=self.input_depth, target=None, label=None, phase=self.phase,
                #                cal_loss=False)
            input_num = len(self.fake_depth)
            indexes = [i for i in range(input_num)]
            # rgb_random_index = random.sample(indexes, int(len(self.fake_rgb) * self.cfg.FAKE_DATA_RATE))
            depth_random_index = random.sample(indexes, int(len(self.fake_depth) * self.cfg.FAKE_DATA_RATE))

            # for i in rgb_random_index:
            #     self.input_rgb[i, :] = self.fake_rgb.data[i, :]
            for j in depth_random_index:
                self.input_depth[j, :] = self.fake_depth.data[j, :]

        self.result = self.net(self.input_rgb, self.input_depth, label=self.label, phase=self.phase, cal_loss=cal_loss)

    def write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES
        task = self.cfg.TASK_TYPE

        if phase == 'train':

            self.writer.add_image(task + '/source_rgb',
                                  torchvision.utils.make_grid(self.input_rgb.data[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            self.writer.add_image(task + '/source_depth',
                                  torchvision.utils.make_grid(self.input_depth.data[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            self.writer.add_scalar(task + '/LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar(task + '/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)

            if 'SEMANTIC' in loss_types:
                self.writer.add_image(task + '/Gen_rgb',
                                      torchvision.utils.make_grid(self.result['gen_rgb'].data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image(task + '/Gen_depth',
                                      torchvision.utils.make_grid(self.result['gen_depth'].data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

        elif phase == 'test':

            self.writer.add_scalar(task + '/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar(task + '/VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].val * 100.0,
                                   global_step=global_step)