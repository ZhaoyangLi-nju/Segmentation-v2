import os
import random
import sys
from datetime import datetime
from functools import reduce

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader  # new add

import util.utils as util
from config.default_config import DefaultConfig
from config.segmentation.resnet_sunrgbd_config import SEG_RESNET_SUNRGBD_CONFIG
from config.segmentation.resnet_cityscape_config import SEG_RESNET_CITYSCAPE_CONFIG
from config.recognition.resnet_sunrgbd_config import REC_RESNET_SUNRGBD_CONFIG
from config.recognition.resnet_nyud2_config import REC_RESNET_NYUD2_CONFIG
from config.recognition.resnet_mit67_config import REC_RESNET_MIT67_CONFIG
from config.infomax.resnet_sunrgbd_config import INFOMAX_RESNET_SUNRGBD_CONFIG
from config.infomax.resnet_nyud2_config import INFOMAX_RESNET_NYUD2_CONFIG
from data import segmentation_dataset_cv2
from data import dataset
from model.trans2_model import Trans2Net
from model.trans2_multimodal import TRans2Multimodal
from model.trans2_infomax import TRans2InfoMax
from model.fusion import Fusion
from model.trans2_infomax_multimodal import TRans2InfoMaxMultimodal

import torch.multiprocessing as mp
import torch.distributed as dist
import apex
import cv2
import torch.nn as nn

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def main():
    cfg = DefaultConfig()
    args = {
        'seg_resnet_sunrgbd': SEG_RESNET_SUNRGBD_CONFIG().args(),
        'seg_resnet_cityscapes': SEG_RESNET_CITYSCAPE_CONFIG().args(),
        'rec_resnet_sunrgbd': REC_RESNET_SUNRGBD_CONFIG().args(),
        'rec_resnet_nyud2': REC_RESNET_NYUD2_CONFIG().args(),
        'rec_resnet_mit67': REC_RESNET_MIT67_CONFIG().args(),
        'infomax_resnet_sunrgbd': INFOMAX_RESNET_SUNRGBD_CONFIG().args(),
        'infomax_resnet_nyud2': INFOMAX_RESNET_NYUD2_CONFIG().args()
    }
    # use shell
    if len(sys.argv) > 1:
        device_ids = torch.cuda.device_count()
        print('device_ids:', device_ids)
        gpu_ids, config_key = sys.argv[1:]
        cfg.parse(args[config_key])
        cfg.GPU_IDS = gpu_ids.split(',')

    else:
        # seg_resnet_sunrgbd
        # seg_resnet_cityscapes
        # infomax_resnet_sunrgbd
        # rec_resnet_sunrgbd
        # rec_resnet_nyud2
        # rec_resnet_mit67
        # infomax_resnet_nyud2
        config_key = 'rec_resnet_sunrgbd'
        cfg.parse(args[config_key])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), cfg.GPU_IDS))

    trans_task = '' + cfg.WHICH_DIRECTION
    if not cfg.NO_TRANS:
        if cfg.MULTI_MODAL:
            trans_task = trans_task + '_multimodal_'

        if 'SEMANTIC' in cfg.LOSS_TYPES or 'PIX2PIX' in cfg.LOSS_TYPES:
            trans_task = trans_task + '_alpha_' + str(cfg.ALPHA_CONTENT)

    evaluate_type = 'sliding_window' if cfg.SLIDE_WINDOWS else 'center_crop'
    log_name = ''.join([cfg.TASK, '_', cfg.ARCH, '_', trans_task, '_', cfg.DATASET])
    cfg.LOG_NAME = ''.join([log_name, '_', '.'.join(cfg.LOSS_TYPES), '_',
                                     evaluate_type,
                                     '_gpus_', str(len(cfg.GPU_IDS)), '_', datetime.now().strftime('%b%d_%H-%M-%S')])
    cfg.LOG_PATH = os.path.join(cfg.LOG_PATH, cfg.MODEL, cfg.LOG_NAME)

    # Setting random seed
    if cfg.MANUAL_SEED is None:
        cfg.MANUAL_SEED = random.randint(1, 10000)
    random.seed(cfg.MANUAL_SEED)
    torch.manual_seed(cfg.MANUAL_SEED)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True

    project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
    print('>>> task path is {0}'.format(project_name))

    util.mkdir('logs')

    # dataset = segmentation_dataset_cv2
    train_transforms = list()
    val_transforms = list()
    ms_targets = []

    train_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
    # train_transforms.append(dataset.RandomScale(cfg.RANDOM_SCALE_SIZE))  #
    # train_transforms.append(dataset.RandomRotate())
    # train_transforms.append(dataset.RandomCrop_Unaligned(cfg.FINE_SIZE, pad_if_needed=True, fill=0))  #
    train_transforms.append(dataset.RandomCrop(cfg.FINE_SIZE, pad_if_needed=True, fill=0))  #
    train_transforms.append(dataset.RandomHorizontalFlip())
    if cfg.TARGET_MODAL == 'lab':
        train_transforms.append(dataset.RGB2Lab())
    if cfg.MULTI_SCALE:
        for item in cfg.MULTI_TARGETS:
            ms_targets.append(item)
        train_transforms.append(dataset.MultiScale(size=cfg.FINE_SIZE,
                                                                scale_times=cfg.MULTI_SCALE_NUM, ms_targets=ms_targets))
    train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
    train_transforms.append(
        dataset.Normalize(mean=cfg.MEAN, std=cfg.STD, ms_targets=ms_targets))

    val_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
    if not cfg.SLIDE_WINDOWS:
        val_transforms.append(dataset.CenterCrop((cfg.FINE_SIZE)))

    if cfg.MULTI_SCALE:
        val_transforms.append(dataset.MultiScale(size=cfg.FINE_SIZE,
                                                                scale_times=cfg.MULTI_SCALE_NUM, ms_targets=ms_targets))
    val_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
    val_transforms.append(dataset.Normalize(mean=cfg.MEAN, std=cfg.STD, ms_targets=ms_targets))

    train_dataset = dataset.__dict__[cfg.DATASET](cfg=cfg, transform=transforms.Compose(train_transforms),
                                                    data_dir=cfg.DATA_DIR_TRAIN, phase_train=True)
    val_dataset = dataset.__dict__[cfg.DATASET](cfg=cfg, transform=transforms.Compose(val_transforms),
                                                  data_dir=cfg.DATA_DIR_VAL, phase_train=False)
    cfg.CLASS_WEIGHTS_TRAIN = train_dataset.class_weights
    cfg.IGNORE_LABEL = train_dataset.ignore_label

    cfg.train_dataset = train_dataset
    cfg.val_dataset = val_dataset

    port = random.randint(8001, 9000)
    ngpus_per_node = len(cfg.GPU_IDS)
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.rank = 0
        cfg.ngpus_per_node = ngpus_per_node
        cfg.dist_url = 'tcp://127.0.0.1:' + str(port)
        cfg.dist_backend = 'nccl'
        cfg.opt_level = 'O0'
        cfg.world_size = 1

    cfg.print_args()

    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.world_size = cfg.ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=cfg.ngpus_per_node, args=(cfg.ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.GPU_IDS, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):

    writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard

    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
        torch.cuda.set_device(gpu)

        # data
        cfg.BATCH_SIZE_TRAIN = int(cfg.BATCH_SIZE_TRAIN / ngpus_per_node)
        cfg.BATCH_SIZE_VAL = int(cfg.BATCH_SIZE_VAL / ngpus_per_node)
        cfg.WORKERS = int(cfg.WORKERS / ngpus_per_node)
        train_sampler = torch.utils.data.distributed.DistributedSampler(cfg.train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(cfg.val_dataset)
        cfg.train_sampler = train_sampler
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(cfg.train_dataset, batch_size=cfg.BATCH_SIZE_TRAIN, shuffle=(train_sampler is None),
                              num_workers=cfg.WORKERS, pin_memory=True, drop_last=True, sampler=train_sampler)

    val_loader = DataLoader(cfg.val_dataset, batch_size=cfg.BATCH_SIZE_VAL, shuffle=False,
                            num_workers=cfg.WORKERS, pin_memory=True, sampler=val_sampler)

    # model
    if cfg.SYNC_BN:
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from lib.sync_bn.modules import BatchNorm2d as SyncBatchNorm
            BatchNorm = SyncBatchNorm
    else:
        BatchNorm = nn.BatchNorm2d

    if cfg.TASK_TYPE == 'infomax':
        model = TRans2InfoMax(cfg, writer=writer)
    elif cfg.MULTI_MODAL:
        model = TRans2Multimodal(cfg, writer=writer)
    elif cfg.TASK_TYPE == 'recognition' and 'fusion' in cfg.MODEL:
        model = Fusion(cfg, writer=writer)
    else:
        model = Trans2Net(cfg, writer=writer, batch_norm=BatchNorm)
    model.set_data_loader(train_loader, val_loader)

    if cfg.RESUME:
        checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
        if cfg.INFERENCE:
            keep_fc = True
        else:
            keep_fc = False
        model.load_checkpoint(checkpoint_path, keep_fc=keep_fc)

    # train
    model.train_parameters(cfg)

    model.save_checkpoint()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
