import math
import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from model import networks

import util.utils as util
from util.average_meter import AverageMeter
from .trans2_model import TRecgNet

class TRecgNet_MULTIMODAL_Conc(TRecgNet):

    def __init__(self, cfg, writer=None):
        super(TRecgNet_MULTIMODAL_Conc, self).__init__(cfg, writer)

    def set_input(self, data):
        self.source_modal = data['image'].to(self.device)
        self.label = data['label'].to(self.device)

        self.target_depth = data['depth'].to(self.device)
        self.target_seg = data['seg'].to(self.device)

    def set_criterion(self, cfg):

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            criterion_segmentation = util.CrossEntropyLoss2d(weight=cfg.CLASS_WEIGHTS_TRAIN,
                                                             ignore_index=cfg.IGNORE_LABEL)
            # criterion_segmentation = util.CrossEntropyLoss2d()
            self.net.set_cls_criterion(criterion_segmentation)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content, in_channel=6).to(self.device)
            self.net.set_content_model(content_model)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            criterion_pix2pix = torch.nn.L1Loss()
            self.net.set_pix2pix_criterion(criterion_pix2pix)

    # encoder-decoder branch
    def _forward(self, iters):

        self.gen = None
        self.source_modal_show = None
        self.target_modal_show = None
        self.cls_loss = None

        if self.phase == 'train':

            if 'CLS' not in self.cfg.LOSS_TYPES:
                if_trans = True
                if_cls = False

            elif self.trans and 'CLS' in self.cfg.LOSS_TYPES:
                if_trans = True
                if_cls = True
            else:
                if_trans = False
                if_cls = True
        else:
            if_cls = True
            if_trans = False
            # for time saving
            if iters > self.cfg.NITER_TOTAL - 500 and 'SEMANTIC' in self.cfg.LOSS_TYPES:
                if_trans = True

        self.source_modal_show = self.source_modal  # rgb

        out_keys = self.build_output_keys(trans=if_trans, cls=if_cls)
        target = torch.cat((self.target_depth, self.target_seg), 1)
        self.result = self.net(source=self.source_modal, target=target,
                               label=self.label, out_keys=out_keys, phase=self.phase)
        # self.result = self.net(source=self.source_modal, target_1=self.target_depth, target_2=self.target_seg,
        #                        label=self.label, out_keys=out_keys, phase=self.phase)

        if if_cls:
            self.cls = self.result['cls']

        if if_trans:
            self.gen = self.result['gen_img']


    def _construct_TRAIN_G_LOSS(self, iters=None):

        loss_total = torch.zeros(1)
        if self.use_gpu:
            loss_total = loss_total.to(self.device)

        if 'CLS' in self.cfg.LOSS_TYPES:
            cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
            loss_total = loss_total + cls_loss

            cls_loss = round(cls_loss.item(), 4)
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss)
            # self.Train_predicted_label = self.cls.data
            self.Train_predicted_label = self.cls.data.max(1)[1].cpu().numpy()

        # ) content supervised
        if 'SEMANTIC' in self.cfg.LOSS_TYPES:

            # content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * (iters / self.cfg.NITER_TOTAL)
            # content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * self.cfg.ALPHA_CONTENT * max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL)
            content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT
            loss_total = loss_total + content_loss

            # content_loss = round(content_loss.item(), 4)
            self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(round(content_loss.item(), 4))

        return loss_total


    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_G_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'VAL_CLS_LOSS',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_IOU',
            'TRAIN_I',
            'TRAIN_U',
            'VAL_I',
            'VAL_U',
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def _write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES

        self.label_show = self.label.data.cpu().numpy()
        self.source_modal_show = self.source_modal

        if phase == 'train':

            self.writer.add_scalar('LR', self.optimizer_ED.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                       global_step=global_step)

            self.writer.add_image('Seg/Train_groundtruth_depth',
                              torchvision.utils.make_grid(self.target_depth[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
            self.writer.add_image('Seg/Train_predicted_depth',
                              torchvision.utils.make_grid(self.gen[:,:3,:,:].data[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
            self.writer.add_image('Seg/Train_groundtruth_seg',
                              torchvision.utils.make_grid(self.target_seg[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
            self.writer.add_image('Seg/Train_predicted_seg',
                              torchvision.utils.make_grid(self.gen[:,3:,:,:].data[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)

            self.writer.add_image('Seg/Train_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if 'CLS' in loss_types:
                self.writer.add_image('Seg/Train_predicted_label',
                                      torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.Train_predicted_label[:6], ignore=self.cfg.IGNORE_LABEL, dataset=self.cfg.DATASET)), 3, normalize=True, range=(0, 255)), global_step=global_step)
                self.writer.add_image('Seg/Train_ground_label',
                                      torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.label_show[:6], ignore=self.cfg.IGNORE_LABEL, dataset=self.cfg.DATASET)), 3, normalize=True, range=(0, 255)), global_step=global_step)

        if phase == 'test':

            self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', float(self.val_iou.mean())*100.0,
                                   global_step=global_step)
