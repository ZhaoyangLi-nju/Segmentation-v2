import os
import random

import h5py
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numbers
from util.utils import color_label_np
import PIL.ImageEnhance as ImageEnhance
from collections import Counter
from skimage import color
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Seg_SUNRGBD(Dataset):

    def __init__(self, cfg=None, transform=None, data_dir=None, phase_train=None):

        self.cfg = cfg
        self.phase_train = phase_train
        self.transform = transform
        self.ignore_label = 255
        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: 0, 2: 1,
                              3: 2, 4: 3, 5: 4, 6: 5,
                              7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,
                              14: 13, 15: 14, 16: 15, 17: 16,
                              18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26,
                              28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36}

        self.img_dir_train_file = './sunrgbd_seg/img_dir_train.txt'
        self.depth_dir_train_file = './sunrgbd_seg/depth_dir_train.txt'
        self.label_dir_train_file = './sunrgbd_seg/label_train.txt'
        self.img_dir_test_file = './sunrgbd_seg/img_dir_test.txt'
        self.depth_dir_test_file = './sunrgbd_seg/depth_dir_test.txt'
        self.label_dir_test_file = './sunrgbd_seg/label_test.txt'

        self.class_weights = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
                              0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
                              2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
                              0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
                              1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
                              4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
                              3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
                              0.750738, 4.040773]

        try:
            with open(self.img_dir_train_file, 'r') as f:
                self.img_dir_train = os.path.join(data_dir, f.read().splitlines())
            with open(self.depth_dir_train_file, 'r') as f:
                self.depth_dir_train = os.path.join(data_dir, f.read().splitlines())
            with open(self.label_dir_train_file, 'r') as f:
                self.label_dir_train = os.path.join(data_dir, f.read().splitlines())
            with open(self.img_dir_test_file, 'r') as f:
                self.img_dir_test = os.path.join(data_dir, f.read().splitlines())
            with open(self.depth_dir_test_file, 'r') as f:
                self.depth_dir_test = os.path.join(data_dir, f.read().splitlines())
            with open(self.label_dir_test_file, 'r') as f:
                self.label_dir_test = os.path.join(data_dir, f.read().splitlines())
        except:

            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(cfg.DATA_DIR, real_dir, 'hha/' + meta.depthname)
                rgb_path = os.path.join(cfg.DATA_DIR, real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(cfg.DATA_DIR, real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(cfg.DATA_DIR, real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)

            local_file_dir = '/'.join(self.img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(self.img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(self.depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(self.label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(self.img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(self.depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(self.label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

        self.seg_dir_train = []
        self.seg_dir_test = []
        try:
            with open(self.img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            for i in range(len(self.img_dir_train)):
                self.img_dir_train[i] = self.img_dir_train[i].split(".")[0] + ".npy"

            with open(self.depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            for i in range(len(self.depth_dir_train)):
                self.depth_dir_train[i] = self.depth_dir_train[i].split(".")[0] + ".npy"

            with open(self.label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            for i in range(len(self.label_dir_train)):
                self.seg_dir_train.append(self.label_dir_train[i].split(".")[0] + "seg.npy")

            with open(self.img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            for i in range(len(self.img_dir_test)):
                self.img_dir_test[i] = self.img_dir_test[i].split(".")[0] + ".npy"

            with open(self.depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            for i in range(len(self.depth_dir_test)):
                self.depth_dir_test[i] = self.depth_dir_test[i].split(".")[0] + ".npy"

            with open(self.label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
            for i in range(len(self.label_dir_test)):
                self.seg_dir_test.append(self.label_dir_test[i].split(".")[0] + "seg.npy")

        except:

            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(real_dir, 'hha/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)

            local_file_dir = '/'.join(self.img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(self.img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(self.depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(self.label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(self.img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(self.depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(self.label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            seg_dir = self.seg_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test
            seg_dir = self.seg_dir_test

        # image = Image.open(img_dir[idx]).convert('RGB')
        # _label = np.load(label_dir[idx])
        # #
        # # image = self.examples[idx]['image']
        # # _label = self.examples[idx]['label']
        # _label_copy = _label.copy()
        # for k, v in self.id_to_trainid.items():
        #     _label_copy[_label == k] = v
        # label = Image.fromarray(_label_copy.astype(np.uint8))
        #
        # depth = None
        # seg = None
        #
        # if self.cfg.MULTI_MODAL:
        #     depth = Image.open(depth_dir[idx]).convert('RGB')
        #     seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
        # elif 'depth' == self.cfg.TARGET_MODAL:
        #     depth = Image.open(depth_dir[idx]).convert('RGB')
        # elif 'seg' == self.cfg.TARGET_MODAL:
        #     seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
        #
        # sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg}
        # for key in list(sample.keys()):
        #     if sample[key] is None:
        #         sample.pop(key)
        #
        # if self.transform:
        #     sample = self.transform(sample)

        image = np.load(img_dir[idx])
        _label = np.load(label_dir[idx])
        _label_copy = _label.copy()
        for k, v in self.id_to_trainid.items():
            _label_copy[_label == k] = v

        depth = None
        seg = None
        lab = None

        if self.cfg.MULTI_MODAL:
            # seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
            depth_array = np.load(depth_dir[idx])
            depth = Image.fromarray(depth_array, mode='RGB')
            seg_array = np.load(seg_dir[idx])
            seg = Image.fromarray(seg_array, mode='RGB')
        elif 'depth' == self.cfg.TARGET_MODAL:
            depth_array = np.load(depth_dir[idx])
            depth = Image.fromarray(depth_array, mode='RGB')
        elif 'seg' == self.cfg.TARGET_MODAL:
            seg_array = np.load(seg_dir[idx])
            seg = Image.fromarray(seg_array, mode='RGB')
        elif 'lab' == self.cfg.TARGET_MODAL:
            lab = color.rgb2lab(image)

        image = Image.fromarray(image, mode='RGB')
        label = Image.fromarray(_label_copy.astype(np.uint8))

        sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg, 'lab': lab}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)

        if self.transform:
            sample = self.transform(sample)

        return sample


class Seg_Cityscapes(torch.utils.data.Dataset):
    def __init__(self, cfg=None, transform=None, data_dir=None, phase_train=True):
        train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
                      "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
                      "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
                      "bremen/", "bochum/", "aachen/"]
        val_dirs = ["frankfurt/", "munster/", "lindau/"]
        test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
        self.id_to_trainid = {-1: -1, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1,
                              9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
                              30: 255, 31: 16, 32: 17, 33: 18}
        # self.class_weights = [2.5965083, 6.7422495, 3.5350077, 9.866795, 9.691752,
        #                       9.369563, 10.289785, 9.954636, 4.308077, 9.491024, 7.6707582, 9.395554,
        #                       10.3475065, 6.3950195, 10.226835, 10.241277, 10.280692, 10.396961, 10.05563]
        self.class_weights = None
        self.cfg = cfg
        self.ignore_label = 255
        self.transform = transform
        if phase_train:
            trainFlag = "train"
            file_dir = train_dirs
        else:
            trainFlag = "val"
            file_dir = val_dirs
        self.img_dir = data_dir + "/leftImg8bit/" + trainFlag + "/"
        self.label_dir = data_dir + "/gtFine/" + trainFlag + "/"
        self.examples = []
        # count=0
        for train_dir in file_dir:
            train_img_dir_path = self.img_dir + train_dir
            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                if 't.npy' not in file_name:
                    continue
                img_id = file_name.split("_leftImg8bit.npy")[0]
                img_path = train_img_dir_path + file_name
                label_img_path = self.label_dir + train_dir + img_id + "_gtFine_labelIds.npy"
                example = {}
                seg_path_npy = label_img_path.split("_gtFine_labelIds.npy")[0] + "_seg.npy"

                example["img_path"] = img_path
                example["label_path"] = label_img_path
                example["seg_path"] = seg_path_npy

                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)
        # if phase_train and self.class_weights == None:
        #     self.class_weights = self.getClassWeight(self.examples)
        #     print(self.class_weights)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example["img_path"]
        label_path = example["label_path"]
        seg_path = example["seg_path"]

        # image = Image.open(img_path).convert('RGB')
        image = np.load(img_path)
        label = np.load(label_path)

        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy.astype(np.uint8))
        image = Image.fromarray(image.astype(np.uint8), mode='RGB')

        seg = None

        if self.cfg.NO_TRANS == False:
            if 'seg' == self.cfg.TARGET_MODAL:
                # seg = Image.fromarray((color_label_np(label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
                seg = np.load(seg_path)
                seg = Image.fromarray(seg.astype(np.uint8), mode='RGB')
            sample = {'image': image, 'label': label, 'seg': seg}
        else:
            # print(image.size)
            sample = {'image': image, 'label': label}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)
        if self.transform:
            sample = self.transform(sample)

        return sample

    # def getClassWeight(self, example):
    #     trainId_to_count = {}
    #     class_weights = []
    #     num_classes = 19
    #     for trainId in range(num_classes):
    #         trainId_to_count[trainId] = 0
    #     for step, samples in enumerate(example):
    #         if step % 100 == 0:
    #             print(step)
    #
    #         # label_img = cv2.imread(label_img_path, -1)
    #         label = np.load(samples["label_img_path"])
    #         label_copy = label.copy()
    #         for k, v in self.id_to_trainid.items():
    #             label_copy[label == k] = v
    #         label_img = Image.fromarray(label_copy.astype(np.uint8))
    #         for trainId in range(num_classes):
    #             # count how many pixels in label_img which are of object class trainId:
    #             trainId_mask = np.equal(label_img, trainId)
    #             trainId_count = np.sum(trainId_mask)
    #             # add to the total count:
    #             trainId_to_count[trainId] += trainId_count
    #     total_count = sum(trainId_to_count.values())
    #     for trainId, count in trainId_to_count.items():
    #         trainId_prob = float(count) / float(total_count)
    #         trainId_weight = 1 / np.log(1.02 + trainId_prob)
    #         class_weights.append(trainId_weight * 0.1)
    #     return class_weights


class Rec_SUNRGBD:
    #bathroom': 0, 'bedroom': 1, 'classroom': 2, 'computer_room': 3, 'conference_room': 4, 'corridor': 5, 'dining_area': 6,
    #'dining_room': 7, 'discussion_area': 8, 'furniture_store': 9, 'home_office': 10, 'kitchen': 11, 'lab': 12, 'lecture_theatre': 13, 
    #'library': 14, 'living_room': 15, 'office': 16, 'rest_space': 17, 'study_space': 18
    def __init__(self, cfg, data_dir=None, transform=None, phase_train=None):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = not cfg.UNLABELED
        self.ignore_label = -100

        if self.labeled:
            self.classes, self.class_to_idx = self.find_classes(self.data_dir)
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
            self.imgs = self.make_dataset()
            self.class_weights = list(Counter([i[1] for i in self.imgs]).values())
        else:
            self.imgs = self.get_images()
            self.class_weights = None

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        # print('**********')
        # print(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # print(class_to_idx)
        # print('**********')
        return classes, class_to_idx

    def make_dataset(self):
        images = []
        dir = os.path.expanduser(self.data_dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    images.append(item)
        return images

    def get_images(self):
        images = []
        dir = os.path.expanduser(self.data_dir)
        image_names = [d for d in os.listdir(dir)]
        for image_name in image_names:
            file = os.path.join(dir, image_name)
            images.append(file)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE[0]:
            A = AB_conc.crop((0, 0, w2, h)).resize(self.cfg.LOAD_SIZE, Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize(self.cfg.LOAD_SIZE, Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'image': A, 'label': label}
        else:
            sample = {'image': A}

        if self.cfg.TARGET_MODAL == 'depth':
            sample['depth'] = B

        if self.cfg.WHICH_DIRECTION == 'BtoA':
            sample['image'], sample['depth'] = sample['depth'], sample['image']

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rec_NYUD2(Rec_SUNRGBD):
    def __init__(self, cfg, data_dir=None, transform=None, phase_train=None):
        super().__init__(cfg, data_dir, transform, phase_train)

# class Rec_MIT67(Rec_SUNRGBD):
#     def __init__(self, cfg, data_dir=None, transform=None, phase_train=None):
#         super().__init__(cfg, data_dir, transform, phase_train)

class Rec_MIT67(Rec_SUNRGBD):

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        sample = {'image': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images


class Resize(transforms.Resize):

    def __call__(self, sample):

        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            if key == 'label':
                sample[key] = F.resize(sample[key], self.size, interpolation=Image.NEAREST)
                continue
            sample[key] = F.resize(sample[key], self.size, interpolation=Image.BICUBIC)

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        img = sample['image']

        # i, j, h, w = self.get_params(image, self.size)

        # if self.padding > 0:
        #     img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        for key in sample.keys():
            # sample[key] = super().__call__(sample[key])
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class RandomCrop_Unaligned(transforms.RandomCrop):

    def __call__(self, sample):
        img = sample['image']

        # i, j, h, w = self.get_params(image, self.size)

        # if self.padding > 0:
        #     img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        for key in sample.keys():
            i, j, h, w = self.get_params(img, self.size)
            # sample[key] = super().__call__(sample[key])
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.center_crop(sample[key], self.size)

        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        if random.random() > 0.5:
            for key in sample.keys():
                if not isinstance(sample[key], Image.Image):
                    continue
                sample[key] = F.hflip(sample[key])

        return sample


class RandomScale(object):

    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image = sample['image']
        target_scale = random.uniform(self.scale_low, self.scale_high)
        target_h = int(round(target_scale * image.size[1]))
        target_w = int(round(target_scale * image.size[0]))

        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            if key == 'label':
                sample['label'] = F.resize(sample['label'], (target_h, target_w), interpolation=Image.NEAREST)
                continue
            sample[key] = F.resize(sample[key], (target_h, target_w), interpolation=Image.BICUBIC)
        return sample


class MultiScale(object):

    def __init__(self, size, scale_times=5, ms_targets=[]):
        assert ms_targets
        self.ms_targets = ms_targets
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_times = scale_times

    def __call__(self, sample):
        h = self.size[0]
        w = self.size[1]

        for key in self.ms_targets:
            if key not in sample.keys():
                raise ValueError('multiscale keys not in sample keys!!!')
            item = sample[key]
            sample[key] = [F.resize(item, (int(h / pow(2, i)), int(w / pow(2, i))), interpolation=Image.BICUBIC) for i in range(self.scale_times)]

        return sample


class RandomRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate=[-10, 10]):
        self.rotate = rotate
        self.p = 0.5

    def __call__(self, sample):
        if random.random() < self.p:
            image, label = sample['image'], sample['label']
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            for key in sample.keys():
                if not isinstance(sample[key], Image.Image):
                    continue
                if key == 'label':
                    sample['label'] = label.rotate(angle, Image.NEAREST)
                    continue
                sample[key] = sample[key].rotate(angle, Image.BILINEAR)
        return sample


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        image = sample['image']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        image = ImageEnhance.Brightness(image).enhance(r_brightness)
        image = ImageEnhance.Contrast(image).enhance(r_contrast)
        image = ImageEnhance.Color(image).enhance(r_saturation)
        sample['image'] = image
        return sample


class ToTensor(object):
    def __init__(self, ms_targets=[]):
        self.ms_targets = ms_targets

    def __call__(self, sample):

        single_targets = list(set(sample.keys()) ^ set(self.ms_targets))

        for key in self.ms_targets:
            sample[key] = [F.to_tensor(item) for item in sample[key]]
        for key in single_targets:
            if not isinstance(sample[key], Image.Image) and key != 'lab':
                continue
            if key == 'label':
                label = sample['label']
                _label = np.maximum(np.array(label, dtype=np.int32), 0)
                sample['label'] = torch.from_numpy(_label).long()
                continue
            sample[key] = F.to_tensor(sample[key])

        return sample


class Normalize(transforms.Normalize):

    def __init__(self, mean, std, ms_targets=[]):
        super().__init__(mean, std)
        self.ms_targets = ms_targets

    def __call__(self, sample):

        single_targets = list(set(sample.keys()) ^ set(self.ms_targets))

        for key in self.ms_targets:
            sample[key] = [F.normalize(item, self.mean, self.std) for item in sample[key]]
        for key in single_targets:
            if key == 'label':
                continue
            sample[key] = F.normalize(sample[key], self.mean, self.std)

        return sample


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""

    def __call__(self, sample):
        image = sample['image']
        img = np.asarray(image, np.uint8)
        lab = color.rgb2lab(img)
        sample['lab'] = lab

        return sample
