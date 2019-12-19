from config.default_config import DefaultConfig
import os

class SEG_RESNET_SUNRGBD_CONFIG:

    def args(self):
        log_dir = os.path.join(DefaultConfig.ROOT_DIR, 'summary')

        ########### Quick Setup ############
        task_type = 'segmentation'
        model = 'FCN_MAXPOOL'
        arch = 'resnet50'
        dataset = 'Seg_SUNRGBD'

        task_name = 'trans2'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'place'
        content_pretrained = 'place'

        multiprocessing = True
        use_apex = True
        sync_bn = True
        gpus = [4,5]
        batch_size_train = 40
        batch_size_val = 40

        base_size = (320, 420)
        load_size = (320, 420)
        random_scale = (0.75, 1.25)
        fine_size = (704, 704)

        niter = 5000
        niter_decay = 10000
        niter_total = niter + niter_decay
        print_freq = niter_total / 100

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        no_trans = False  # if True, no translation loss
        if no_trans:
            loss = ['CLS']
            target_modal = None
            multi_modal = False
        else:
            loss = ['CLS', 'SEMANTIC']
            target_modal = 'depth'
            # target_modal = 'seg'
            multi_modal = False

        base_size = (256, 256)
        load_size = (256, 256)
        fine_size = (224, 224)
        lr = 2e-4
        filters = 'bottleneck'

        evaluate = True  # report mean acc after each epoch
        slide_windows = False

        unlabeld = False  # True for training with unlabeled data
        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.5
        which_content_net = 'resnet50'

        multi_scale = False
        multi_targets = ['depth']
        # multi_targets = ['seg']
        which_score = 'up'
        norm = 'in'

        resume = False
        resume_path = 'FCN/2019_09_17_13_50_34/FCN_AtoB_5000.pth'

        return {

            'TASK_TYPE': task_type,
            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,
            'PRETRAINED': pretrained,
            'FILTERS': filters,
            'DATASET': dataset,
            'MEAN': mean,
            'STD': std,

            'LOG_PATH': log_dir,
            'DATA_DIR_TRAIN': '/home/lzy/lzy/dataset/sunrgbd_seg',
            'DATA_DIR_VAL': '/home/lzy/lzy/dataset/sunrgbd_seg',
            # 'DATA_DIR': DefaultConfig.ROOT_DIR + '/datasets/vm_data/sunrgbd_seg',

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': True,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 37,
            'UNLABELED': unlabeld,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'BASE_SIZE': base_size,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,

            'LR': lr,
            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'SLIDE_WINDOWS': slide_windows,
            'PRINT_FREQ': print_freq,

            'MULTIPROCESSING_DISTRIBUTED': multiprocessing,
            'USE_APEX': use_apex,
            'SYNC_BN': sync_bn,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }
