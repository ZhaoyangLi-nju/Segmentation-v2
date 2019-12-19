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


class TRans2InfoMax(Trans2Net):

    def __init__(self, cfg, writer=None):
        super(TRans2InfoMax, self).__init__(cfg, writer)

    def _define_networks(self):

        self.net = networks.Source_Model(self.cfg, device=self.device)
        self.cross_encoder = networks.Cross_Model(self.cfg, device=self.device)
        self.d_distribute = networks.GANDiscriminator(self.cfg, device=self.device)
        self.model_names = ['net', 'cross_encoder', 'd_distribute']

        networks.print_network(self.net)
        networks.print_network(self.cross_encoder)
        networks.print_network(self.d_distribute)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            criterion_pix2pix = torch.nn.L1Loss()
            self.cross_encoder.set_pix2pix_criterion(criterion_pix2pix)

    def set_device(self):

        if not self.cfg.MULTIPROCESSING_DISTRIBUTED:
            self.net = nn.DataParallel(self.net).to(self.device)
            self.cross_encoder = nn.DataParallel(self.cross_encoder).to(self.device)
            self.d_distribute = nn.DataParallel(self.d_distribute).to(self.device)

    def set_optimizer(self, cfg):

        self.optimizers = []

        # if self.cfg.RESUME:
        #     self.params_list = []
        #     self.modules_ft = [self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        #     self.modules_sc = [self.net.evaluator]
        #
        #
        #     for module in self.modules_ft:
        #         self.params_list.append(dict(params=module.parameters(), lr=cfg.LR))
        #     for module in self.modules_sc:
        #         self.params_list.append(dict(params=module.parameters(), lr=cfg.LR * 10))
        #     self.optimizer_g = torch.optim.Adam(self.params_list, lr=cfg.LR, betas=(0.5, 0.999))
        # else:
        self.optimizer_g = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        self.optimizer_c = torch.optim.Adam(self.cross_encoder.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.SGD(self.d_distribute.parameters(), lr=cfg.LR, momentum=0.9, weight_decay=0.0005)

        if cfg.MULTIPROCESSING_DISTRIBUTED:
            if cfg.USE_APEX:
                self.net, self.optimizer_g = apex.amp.initialize(self.net.cuda(), self.optimizer_g, opt_level=cfg.opt_level)
                self.cross_encoder, self.optimizer_c = apex.amp.initialize(self.cross_encoder.cuda(), self.optimizer_c, opt_level=cfg.opt_level)
                self.d_distribute, self.optimizer_d = apex.amp.initialize(self.d_distribute.cuda(), self.optimizer_d, opt_level=cfg.opt_level)
                self.net = DDP(self.net)
                self.cross_encoder = DDP(self.cross_encoder)
                self.d_distribute = DDP(self.d_distribute)
            else:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net.cuda(), device_ids=[cfg.gpu])
                self.cross_encoder = torch.nn.parallel.DistributedDataParallel(self.cross_encoder.cuda(), device_ids=[cfg.gpu])
                self.d_distribute = torch.nn.parallel.DistributedDataParallel(self.d_distribute.cuda(), device_ids=[cfg.gpu])

        self.optimizers.append(self.optimizer_d)
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_c)

    # def get_patch(self, img):
    #
    #     # Input of the function is a tensor [B, C, H, W]
    #     # Output of the functions is a tensor [B * 49, C, 64, 64]
    #
    #     patch_batch = None
    #     all_patches_list = []
    #
    #     for y_patch in range(3):
    #         for x_patch in range(3):
    #             y1 = y_patch * 64
    #             y2 = y1 + 128
    #
    #             x1 = x_patch * 64
    #             x2 = x1 + 128
    #
    #             img_patches = img[:, :, y1:y2, x1:x2]  # Batch(img_idx in batch), channels xrange, yrange
    #             img_patches = img_patches.unsqueeze(dim=1)
    #             all_patches_list.append(img_patches)
    #
    #             # print(patch_batch.shape)
    #     all_patches_tensor = torch.cat(all_patches_list, dim=1)
    #
    #     patches_per_image = []
    #     for b in range(all_patches_tensor.shape[0]):
    #         patches_per_image.append(all_patches_tensor[b])
    #
    #     patch_batch = torch.cat(patches_per_image, dim=0)
    #     return patch_batch

    # encoder-decoder branch
    def _forward(self, class_only=False):

        # if self.phase == 'train':
        #     self.source_modal = self.get_patch(self.source_modal)
        #     self.target_modal = self.get_patch(self.target_modal)
        #
        #     self.source_modal.view(self.batch_size, 3, 3, -1)

        if self.label is not None:
            label = self.label
        else:
            label = None
        self.result_g = self.net(self.source_modal, target=self.target_modal, label=label, class_only=class_only)
        if self.phase == 'train' and not self.cfg.NO_TRANS:
            self.result_c = self.cross_encoder(self.result_g['gen_cross'], target=self.target_modal)

    def _optimize(self, iter):

        self._forward()

        if 'GAN' in self.cfg.LOSS_TYPES:

            self.set_requires_grad([self.cross_encoder, self.net], False)
            self.set_requires_grad(self.d_distribute, True)
            fake_d = torch.cat((self.result_c['feat_gen'], self.result_c['feat_target']), 1)
            real_d = torch.cat((self.result_c['feat_target'], self.result_c['feat_target']), 1)
            # fake_d = self.result_c['feat_gen']
            # real_d = self.result_c['feat_target']

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                loss_d_fake = self.d_distribute(fake_d.detach(), False)
                loss_d_true = self.d_distribute(real_d.detach(), True)
            else:
                loss_d_fake = self.d_distribute(fake_d.detach(), False).mean()
                loss_d_true = self.d_distribute(real_d.detach(), True).mean()

            loss_d = (loss_d_fake + loss_d_true) * 0.5
            self.loss_meters['TRAIN_GAN_D_LOSS'].update(loss_d.item(), self.batch_size)

            self.optimizer_d.zero_grad()
            if self.cfg.USE_APEX and self.cfg.MULTIPROCESSING_DISTRIBUTED:
                with apex.amp.scale_loss(loss_d, self.optimizer_d) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d.backward()
            self.optimizer_d.step()

        # G
        loss_g = self._construct_loss(iter)
        self.set_requires_grad([self.cross_encoder, self.net], True)
        if self.d_distribute is not None:
            self.set_requires_grad(self.d_distribute, False)

        self.optimizer_c.zero_grad()
        self.optimizer_g.zero_grad()
        if self.cfg.USE_APEX and self.cfg.MULTIPROCESSING_DISTRIBUTED:
            with apex.amp.scale_loss(loss_g, [self.optimizer_c, self.optimizer_g]) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_g.backward()
        self.optimizer_c.step()
        self.optimizer_g.step()

    def _construct_loss(self, iter=None):

        loss_total = torch.zeros(1).to(self.device)
        # decay_coef = 1
        decay_coef = (iter / self.cfg.NITER_TOTAL)  # small to big

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                local_loss = self.result_c['pix2pix_loss'] * self.cfg.ALPHA_LOCAL
                loss_total += local_loss
            else:
                local_loss = self.result_c['pix2pix_loss'].mean() * self.cfg.ALPHA_LOCAL

            self.loss_meters['TRAIN_PIX2PIX_LOSS'].update(local_loss.item(), self.batch_size)

        if 'PRIOR' in self.cfg.LOSS_TYPES:
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                prior_loss = self.result_g['prior_loss'] * self.cfg.ALPHA_PRIOR
                loss_total += prior_loss
            else:
                prior_loss = self.result_g['prior_loss'].mean() * self.cfg.ALPHA_PRIOR

            self.loss_meters['TRAIN_PRIOR_LOSS'].update(prior_loss.item(), self.batch_size)

        if 'CROSS' in self.cfg.LOSS_TYPES:
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:

                cross_loss = self.result_c['cross_loss'] * self.cfg.ALPHA_CROSS

                # cross_loss_self = self.result_c['cross_loss_self'] * self.cfg.ALPHA_CROSS * 0.2
            else:
                cross_loss = self.result_c['cross_loss'].mean() * self.cfg.ALPHA_CROSS
                # cross_loss_self = self.result_c['cross_loss_self'].mean() * self.cfg.ALPHA_CROSS * decay_coef

            loss_total += cross_loss
            # loss_total += cross_loss_self
            # loss_total += cross_loss

            self.loss_meters['TRAIN_CROSS_LOSS'].update(cross_loss.item(), self.batch_size)
            # self.loss_meters['TRAIN_CROSS_LOSS_SELF'].update(cross_loss_self.item(), self.batch_size)

        if 'HOMO' in self.cfg.LOSS_TYPES:
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                homo_loss = self.result_g['homo_loss'] * self.cfg.ALPHA_CROSS
                loss_total += homo_loss
            else:
                homo_loss = self.result_g['homo_loss'].mean() * self.cfg.ALPHA_CROSS

            self.loss_meters['TRAIN_HOMO_LOSS'].update(homo_loss.item(), self.batch_size)

        if 'CLS' in self.cfg.LOSS_TYPES:

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                cls_loss = self.result_g['cls_loss']
            else:
                cls_loss = self.result_g['cls_loss'].mean()
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss.item(), self.batch_size)

            loss_total += cls_loss

        if 'GAN' in self.cfg.LOSS_TYPES:

            # real_g = self.result_c['feat_gen']
            real_g = torch.cat((self.result_c['feat_gen'], self.result_c['feat_target']), 1)
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                loss_gan_g = self.d_distribute(real_g, True) * self.cfg.ALPHA_GAN
            else:
                loss_gan_g = self.d_distribute(real_g, True).mean() * self.cfg.ALPHA_GAN
            self.loss_meters['TRAIN_GAN_G_LOSS'].update(loss_gan_g.item(), self.batch_size)

            loss_total += loss_gan_g

        return loss_total

    def set_log_data(self, cfg):

        super().set_log_data(cfg)
        self.log_keys = [
            'TRAIN_CROSS_LOSS',
            'TRAIN_CROSS_LOSS_SELF',
            'TRAIN_HOMO_LOSS',
            'TRAIN_LOCAL_LOSS',
            'TRAIN_PRIOR_LOSS',
            'INTERSECTION_MLP',
            'LABEL_MLP',
            'INTERSECTION_LIN',
            'LABEL_LIN',
            'VAL_CLS_ACC_MLP',
            'VAL_CLS_MEAN_ACC_MLP',
            'TRAIN_GAN_D_LOSS',
            'TRAIN_GAN_G_LOSS',
            'TRAIN_CONTRASTIVE_LOSS'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def evaluate(self):

        # evaluate model on test_loader
        self.net.eval()
        self.phase = 'test'

        intersection_meter_mlp = self.loss_meters['INTERSECTION_MLP']
        target_meter_mlp = self.loss_meters['LABEL_MLP']

        for i, data in enumerate(self.val_loader):
            self.set_input(data)
            with torch.no_grad():
                self._forward(class_only=True)

                pred = self.result_g['pred'].data.max(1)[1]
                # lgt_glb_mlp = lgt_glb_mlp
                # lgt_glb_lin = lgt_glb_lin.data.max(1)[1]
                # [lgt_glb_mlp, lgt_glb_lin] = self.result_g['pred']
                # lgt_glb_mlp = lgt_glb_mlp.data.max(1)[1]
                # lgt_glb_lin = lgt_glb_lin.data.max(1)[1]

            intersection_mlp, union_mlp, label_mlp = util.intersectionAndUnionGPU(pred, self.label,
                                                                      self.cfg.NUM_CLASSES)
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                dist.all_reduce(intersection_mlp), dist.all_reduce(union_mlp), dist.all_reduce(label_mlp)

            intersection_mlp, union_mlp, label_mlp = intersection_mlp.cpu().numpy(), union_mlp.cpu().numpy(), label_mlp.cpu().numpy()

            intersection_meter_mlp.update(intersection_mlp, self.batch_size)
            target_meter_mlp.update(label_mlp, self.batch_size)

        # Mean ACC
        allAcc_mlp = sum(intersection_meter_mlp.sum) / (sum(target_meter_mlp.sum) + 1e-10)
        accuracy_class_mlp = intersection_meter_mlp.sum / (target_meter_mlp.sum + 1e-10)
        mAcc_mlp = np.mean(accuracy_class_mlp)
        self.loss_meters['VAL_CLS_ACC_MLP'].update(allAcc_mlp)
        self.loss_meters['VAL_CLS_MEAN_ACC_MLP'].update(mAcc_mlp)

    def write_loss(self, phase, global_step):

        task = self.cfg.TASK_TYPE
        self.writer.add_image(task + '/rgb',
                              torchvision.utils.make_grid(self.source_modal[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
        if phase == 'train':

            if not self.cfg.NO_TRANS:

                for k, v in self.result_g.items():
                    if 'gen' in k:
                        # if isinstance(self.result_g[k], list):
                        #     for i, (gen, _depth) in enumerate(zip(self.result_g['gen'], self.target_modal)):
                        #         self.writer.add_image(task + '/' + k + str(self.cfg.FINE_SIZE[0] / pow(2, i)),
                        #                          torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                        #                                                      normalize=True),
                        #                          global_step=global_step)
                        #         self.writer.add_image(task + '/target' + str(self.cfg.FINE_SIZE[0] / pow(2, i)),
                        #                          torchvision.utils.make_grid(_depth[:6].clone().cpu().data, 3,
                        #                                                      normalize=True),
                        #                          global_step=global_step)
                        # else:
                        self.writer.add_image(task + '/' + k,
                                              torchvision.utils.make_grid(
                                                  self.result_g[k][:6].clone().cpu().data, 3,
                                                  normalize=True), global_step=global_step)

                self.writer.add_image(task + '/target',
                                      torchvision.utils.make_grid(self.target_modal[:6].clone().cpu().data, 3,

                                                                  normalize=True), global_step=global_step)
                # self.writer.add_image(task + '/target_neg',
                #                       torchvision.utils.make_grid(self.target_modal_neg[:6].clone().cpu().data, 3,
                #
                #                                                   normalize=True), global_step=global_step)

            self.writer.add_scalar(task + '/LR', self.optimizer_g.param_groups[0]['lr'],
                                   global_step=global_step)

            for k, v in self.loss_meters.items():
                if 'LOSS' in k and v.avg > 0:
                    self.writer.add_scalar(task + '/' + k, v.avg,
                                           global_step=global_step)

        elif phase == 'test':

            for k, v in self.loss_meters.items():
                if ('MEAN' in k or 'ACC' in k) and v.val > 0:
                    self.writer.add_scalar(task + '/' + k, v.val * 100.0,
                                           global_step=global_step)