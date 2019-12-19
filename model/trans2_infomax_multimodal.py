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


class TRans2InfoMaxMultimodal(Trans2Net):

    def __init__(self, cfg, writer=None):
        super(TRans2InfoMaxMultimodal, self).__init__(cfg, writer)

    def _define_networks(self):

        self.net = networks.Infomax_Homo_And_Cross(cfg=self.cfg, device=self.device)
        self.infomaxer = networks.Contrastive_CrossModal(self.cfg, device=self.device)

        networks.print_network(self.net)
        networks.print_network(self.infomaxer)

    def set_device(self):

        if not self.cfg.MULTIPROCESSING_DISTRIBUTED:
            self.net = nn.DataParallel(self.net).to(self.device)
            self.infomaxer = nn.DataParallel(self.infomaxer).to(self.device)

    def set_optimizer(self, cfg):

        self.optimizers = []

        self.optimizer_encoder = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        self.optimizer_infomaxer = torch.optim.Adam(self.infomaxer.parameters(), lr=cfg.LR, betas=(0.5, 0.999))

        if cfg.MULTIPROCESSING_DISTRIBUTED:
            if cfg.USE_APEX:
                self.net, self.optimizer_encoder = apex.amp.initialize(self.net.cuda(), self.optimizer_encoder, opt_level=cfg.opt_level)
                self.infomaxer, self.optimizer_infomaxer = apex.amp.initialize(self.infomaxer.cuda(), self.optimizer_infomaxer, opt_level=cfg.opt_level)
                self.net = apex.parallel.DistributedDataParallel(self.net)
            else:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net.cuda(), device_ids=[cfg.gpu])

        self.optimizers.append(self.optimizer_encoder)
        self.optimizers.append(self.optimizer_infomaxer)

    # encoder-decoder branch
    def _forward(self, class_only=False):
        self.result_encoder = self.net(self.source_modal, label=self.label, class_only=class_only)

    def _optimize(self, iter):

        self._forward()

        self.result_infomaxer = self.infomaxer(self.result_encoder, target=self.target_modal, label=self.label)

        total_loss = self._construct_loss()

        if self.cfg.MULTIPROCESSING_DISTRIBUTED:
            cls_loss = self.result_encoder['cls_loss']
        else:
            cls_loss = self.result_encoder['cls_loss'].mean()
        self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss.item(), self.batch_size)

        total_loss = total_loss + cls_loss

        self.optimizer_encoder.zero_grad()
        self.optimizer_infomaxer.zero_grad()
        if self.cfg.USE_APEX and self.cfg.MULTIPROCESSING_DISTRIBUTED:
            with apex.amp.scale_loss(total_loss, self.optimizer_infomaxer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_infomaxer.step()


    def _construct_loss(self, iters=None):

        loss_total = torch.zeros(1).to(self.device)

        if 'LOCAL' in self.cfg.LOSS_TYPES:

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                local_loss = self.result_infomaxer['local_loss'] * self.cfg.ALPHA_LOCAL
                loss_total += local_loss
            else:
                local_loss = self.result_infomaxer['local_loss'].mean() * self.cfg.ALPHA_LOCAL

            self.loss_meters['TRAIN_LOCAL_LOSS'].update(local_loss.item(), self.batch_size)

        if 'PRIOR' in self.cfg.LOSS_TYPES:
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                prior_loss = self.result_infomaxer['prior_loss'] * self.cfg.ALPHA_PRIOR
                loss_total += prior_loss
            else:
                prior_loss = self.result_infomaxer['prior_loss'].mean() * self.cfg.ALPHA_PRIOR

            self.loss_meters['TRAIN_PRIOR_LOSS'].update(prior_loss.item(), self.batch_size)

        if 'CROSS' in self.cfg.LOSS_TYPES:
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                cross_loss = self.result_infomaxer['cross_loss'] * self.cfg.ALPHA_CROSS
                loss_total += cross_loss
            else:
                cross_loss = self.result_infomaxer['cross_loss'].mean() * self.cfg.ALPHA_CROSS

            self.loss_meters['TRAIN_CROSS_LOSS'].update(cross_loss.item(), self.batch_size)

        if 'HOMO' in self.cfg.LOSS_TYPES:
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                homo_loss = self.result_infomaxer['homo_loss'] * self.cfg.ALPHA_CROSS
                loss_total += homo_loss
            else:
                homo_loss = self.result_infomaxer['homo_loss'].mean() * self.cfg.ALPHA_CROSS

            self.loss_meters['TRAIN_HOMO_LOSS'].update(homo_loss.item(), self.batch_size)

        return loss_total

    def set_log_data(self, cfg):

        super().set_log_data(cfg)
        self.log_keys = [
            'TRAIN_CROSS_LOSS',
            'TRAIN_HOMO_LOSS',
            'TRAIN_LOCAL_LOSS',
            'TRAIN_PRIOR_LOSS',
            'INTERSECTION_MLP',
            'LABEL_MLP',
            'INTERSECTION_LIN',
            'LABEL_LIN',
            'VAL_CLS_ACC_LIN',
            'VAL_CLS_MEAN_ACC_LIN',
            'VAL_CLS_ACC_MLP',
            'VAL_CLS_MEAN_ACC_MLP'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def evaluate(self):

        # evaluate model on test_loader
        self.net.eval()
        self.phase = 'test'

        intersection_meter_mlp = self.loss_meters['INTERSECTION_MLP']
        target_meter_mlp = self.loss_meters['LABEL_MLP']
        intersection_meter_lin = self.loss_meters['INTERSECTION_LIN']
        target_meter_lin = self.loss_meters['LABEL_LIN']

        for i, data in enumerate(self.val_loader):
            self.set_input(data)
            with torch.no_grad():
                self._forward(class_only=True)

                [lgt_glb_mlp, lgt_glb_lin] = self.result_encoder['pred']
                lgt_glb_mlp = lgt_glb_mlp.data.max(1)[1]
                lgt_glb_lin = lgt_glb_lin.data.max(1)[1]

            intersection_mlp, union_mlp, label_mlp = util.intersectionAndUnionGPU(lgt_glb_mlp, self.label,
                                                                      self.cfg.NUM_CLASSES)
            intersection_lin, union_lin, label_lin = util.intersectionAndUnionGPU(lgt_glb_lin, self.label,
                                                                      self.cfg.NUM_CLASSES)
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                dist.all_reduce(intersection_mlp), dist.all_reduce(union_mlp), dist.all_reduce(label_mlp)
                dist.all_reduce(intersection_lin), dist.all_reduce(union_lin), dist.all_reduce(label_lin)

            intersection_mlp, union_mlp, label_mlp = intersection_mlp.cpu().numpy(), union_mlp.cpu().numpy(), label_mlp.cpu().numpy()
            intersection_lin, union_lin, label_lin = intersection_lin.cpu().numpy(), union_lin.cpu().numpy(), label_lin.cpu().numpy()

            intersection_meter_mlp.update(intersection_mlp, self.batch_size)
            target_meter_mlp.update(label_mlp, self.batch_size)

            intersection_meter_lin.update(intersection_lin, self.batch_size)
            target_meter_lin.update(label_lin, self.batch_size)


        # Mean ACC
        allAcc_mlp = sum(intersection_meter_mlp.sum) / (sum(target_meter_mlp.sum) + 1e-10)
        accuracy_class_mlp = intersection_meter_mlp.sum / (target_meter_mlp.sum + 1e-10)
        mAcc_mlp = np.mean(accuracy_class_mlp)
        self.loss_meters['VAL_CLS_ACC_MLP'].update(allAcc_mlp)
        self.loss_meters['VAL_CLS_MEAN_ACC_MLP'].update(mAcc_mlp)

        allAcc_lin = sum(intersection_meter_lin.sum) / (sum(target_meter_lin.sum) + 1e-10)
        accuracy_class_lin = intersection_meter_lin.sum / (target_meter_lin.sum + 1e-10)
        mAcc_lin = np.mean(accuracy_class_lin)
        self.loss_meters['VAL_CLS_ACC_LIN'].update(allAcc_lin)
        self.loss_meters['VAL_CLS_MEAN_ACC_LIN'].update(mAcc_lin)

    def write_loss(self, phase, global_step):

        task = self.cfg.TASK_TYPE
        self.writer.add_image(task + '/rgb',
                              torchvision.utils.make_grid(self.source_modal[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
        if phase == 'train':
            self.writer.add_scalar(task + '/LR', self.optimizer_infomaxer.param_groups[0]['lr'], global_step=global_step)
            self.writer.add_scalar(task + '/cross_loss', self.loss_meters['TRAIN_CROSS_LOSS'].avg, global_step=global_step)
            self.writer.add_scalar(task + '/local_loss', self.loss_meters['TRAIN_LOCAL_LOSS'].avg, global_step=global_step)
            self.writer.add_scalar(task + '/prior_loss', self.loss_meters['TRAIN_PRIOR_LOSS'].avg, global_step=global_step)

            if not self.cfg.NO_TRANS:
                # if isinstance(self.result_encoder['gen'], list):
                #     for i, (gen, _depth) in enumerate(zip(self.result_encoder['gen'], self.target_modal)):
                #         self.writer.add_image(task + '/Gen' + str(self.cfg.FINE_SIZE[0] / pow(2, i)),
                #                          torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                #                                                      normalize=True),
                #                          global_step=global_step)
                #         self.writer.add_image(task + '/target' + str(self.cfg.FINE_SIZE[0] / pow(2, i)),
                #                          torchvision.utils.make_grid(_depth[:6].clone().cpu().data, 3,
                #                                                      normalize=True),
                #                          global_step=global_step)
                # else:
                for k, v in self.result_encoder.items():
                    if 'gen' in k:
                        self.writer.add_image(task + '/' + k,
                                              torchvision.utils.make_grid(
                                                  v[:6].clone().cpu().data, 3,
                                                  normalize=True), global_step=global_step)

                self.writer.add_image(task + '/target',
                                      torchvision.utils.make_grid(self.target_modal[:6].clone().cpu().data, 3,

                                                                  normalize=True), global_step=global_step)
        elif phase == 'test':

            self.writer.add_scalar(task + '/VAL_CLS_ACC_MLP', self.loss_meters['VAL_CLS_ACC_MLP'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar(task + '/VAL_CLS_MEAN_ACC_MLP', self.loss_meters['VAL_CLS_MEAN_ACC_MLP'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar(task + '/VAL_CLS_ACC_LIN', self.loss_meters['VAL_CLS_ACC_LIN'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar(task + '/VAL_CLS_MEAN_ACC_LIN', self.loss_meters['VAL_CLS_MEAN_ACC_LIN'].val * 100.0,
                                   global_step=global_step)