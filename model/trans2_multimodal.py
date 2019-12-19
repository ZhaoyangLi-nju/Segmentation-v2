import torch
import torchvision

from util.average_meter import AverageMeter
from .trans2_model import Trans2Net


class TRans2Multimodal(Trans2Net):

    def __init__(self, cfg, writer=None):
        super(TRans2Multimodal, self).__init__(cfg, writer)

    def set_input(self, data):
        super().set_input(data)
        self.target_depth = data['depth'].to(self.device)
        self.target_seg = data['seg'].to(self.device)

    # encoder-decoder branch
    def _forward(self, cal_loss=True):

        self.result = self.net(source=self.source_modal, target_1=self.target_depth, target_2=self.target_seg,
                               label=self.label, phase=self.phase, cal_loss=cal_loss)

    def _construct_loss(self, iters=None):

        loss_total = torch.zeros(1).to(self.device)

        if 'CLS' in self.cfg.LOSS_TYPES:
            cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
            loss_total = loss_total + cls_loss

            cls_loss = round(cls_loss.item(), 4)
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss)

        # ) content supervised
        if 'SEMANTIC' in self.cfg.LOSS_TYPES:

            self.gen = [self.result['gen_depth'], self.result['gen_seg']]
            content_loss_depth = self.result['loss_content_depth'].mean() * self.cfg.ALPHA_CONTENT
            content_loss_seg = self.result['loss_content_seg'].mean() * self.cfg.ALPHA_CONTENT
            content_loss = content_loss_depth + content_loss_seg
            loss_total = loss_total + content_loss

            # content_loss = round(content_loss.item(), 4)
            self.loss_meters['TRAIN_SEMANTIC_LOSS_2DEPTH'].update(round(content_loss_depth.item(), 4))
            self.loss_meters['TRAIN_SEMANTIC_LOSS_2SEG'].update(round(content_loss_seg.item(), 4))

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:

            pix2pix_loss_depth = self.result['loss_pix2pix_depth'].mean()
            pix2pix_loss_seg = self.result['loss_pix2pix_seg'].mean()
            pix2pix_loss = pix2pix_loss_depth + pix2pix_loss_seg
            loss_total = loss_total + pix2pix_loss

            # content_loss = round(content_loss.item(), 4)
            self.loss_meters['TRAIN_PIX2PIX_LOSS_2DEPTH'].update(round(pix2pix_loss_depth.item(), 4))
            self.loss_meters['TRAIN_PIX2PIX_LOSS_2SEG'].update(round(pix2pix_loss_seg.item(), 4))

        return loss_total


    def set_log_data(self, cfg):

        super().set_log_data(cfg)
        self.log_keys = [
            'TRAIN_SEMANTIC_LOSS_2DEPTH',
            'TRAIN_SEMANTIC_LOSS_2SEG',
            'TRAIN_PIX2PIX_LOSS_2DEPTH',
            'TRAIN_PIX2PIX_LOSS_2SEG',
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES
        super().write_loss(phase, global_step)

        if phase == 'train':

            if 'SEMANTIC' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS_2DEPTH', self.loss_meters['TRAIN_SEMANTIC_LOSS_2DEPTH'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS_2SEG', self.loss_meters['TRAIN_SEMANTIC_LOSS_2SEG'].avg,
                                       global_step=global_step)

            if 'PIX2PIX' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_PIX2PIX_LOSS_2DEPTH', self.loss_meters['TRAIN_PIX2PIX_LOSS_2DEPTH'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('Seg/TRAIN_PIX2PIX_LOSS_2SEG', self.loss_meters['TRAIN_PIX2PIX_LOSS_2SEG'].avg,
                                       global_step=global_step)

            self.writer.add_image('Seg/Train_groundtruth_depth',
                              torchvision.utils.make_grid(self.target_depth[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
            self.writer.add_image('Seg/Train_predicted_depth',
                              torchvision.utils.make_grid(self.gen[0].data[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
            self.writer.add_image('Seg/Train_groundtruth_seg',
                              torchvision.utils.make_grid(self.target_seg[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)
            self.writer.add_image('Seg/Train_predicted_seg',
                              torchvision.utils.make_grid(self.gen[1].data[:6].clone().cpu().data, 3,
                                                          normalize=True), global_step=global_step)