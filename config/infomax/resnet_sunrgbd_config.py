from config.default_config import DefaultConfig

class INFOMAX_RESNET_SUNRGBD_CONFIG:

    def args(self):
        log_dir = DefaultConfig.ROOT_DIR + '/summary/'

        ########### Quick Setup ############
        task_type = 'infomax'
        model = 'infomax'
        arch = 'resnet18'
        dataset = 'Rec_SUNRGBD'

        task_name = 'test'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = ''
        which_direction = ''

        multiprocessing = True
        use_apex = True
        sync_bn = True
        gpus = [7]  # 0, 1, 2, 3, 4, 5, 6, 7
        batch_size_train = 128
        batch_size_val = 128

        niter = 4000
        niter_decay = 10000
        niter_total = niter + niter_decay
        print_freq = niter_total / 100

        no_trans = False  # if True, no translation loss
        if no_trans:
            loss = ['CLS']
        else:
            loss = ['CROSS', 'GAN', 'CLS']
        target_modal = 'depth'

        unlabeled = False
        is_finetune = False  # if True, finetune the backbone with downstream tasks
        evaluate = False  # report mean acc after each epoch
        resume = False
        resume_path = 'infomax/infomax_pre_diml_10000_Nov15_16-21.pth'

        multi_scale = False
        multi_scale_num = 1
        multi_targets = ['lab']

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        base_size = (256, 256)
        load_size = (256, 256)
        fine_size = (224, 224)
        lr = 4e-4
        filters = 'bottleneck'
        norm = 'bn'
        slide_windows = False

        alpha_local = 1
        alpha_prior = 0.05
        alpha_cross = 10
        alpha_gan = 1

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
            'UNLABELED': unlabeled,
            'DATA_DIR_TRAIN': '/data/dudapeng/datasets/sun_rgbd/conc_jet_labeled/train',
            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/nyud2/conc_data/10k_conc',
            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/traintest6/',
            # 'DATA_DIR_TRAIN': '/data0/dudapeng/workspace/datasets/nyud2/conc_data/10k_conc',

            # 'DATA_DIR_TRAIN': '/data0/dudapeng/workspace/datasets/sun_rgbd/data_in_class_mix/conc_data/train',
            # 'DATA_DIR_VAL': '/data0/dudapeng/workspace/datasets/sun_rgbd/data_in_class_mix/conc_data/test',
            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/train',
            'DATA_DIR_VAL': '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/test',

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': False,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,
            'WHICH_DIRECTION': which_direction,

            #### DATA
            'NUM_CLASSES': 19,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'BASE_SIZE': base_size,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,
            'FT': is_finetune,

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
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_SCALE_NUM': multi_scale_num,
            'MULTI_TARGETS': multi_targets,
            'ALPHA_LOCAL': alpha_local,
            'ALPHA_PRIOR': alpha_prior,
            'ALPHA_CROSS': alpha_cross,
            'ALPHA_GAN': alpha_gan,
            'UPSAMPLE_NORM': norm
        }

