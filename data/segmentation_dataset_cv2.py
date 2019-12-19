import os
import random

import h5py
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numbers
from util.utils import color_label_np
import PIL.ImageEnhance as ImageEnhance
from skimage import io
import collections
import cv2

img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
img_dir_test_file = './data/img_dir_test.txt'
depth_dir_test_file = './data/depth_dir_test.txt'
label_dir_test_file = './data/label_test.txt'

class SUNRGBD(Dataset):
    def __init__(self, cfg=None, transform=None, phase_train=True, data_dir=None):

        self.cfg = cfg
        self.phase_train = phase_train
        self.transform = transform
        self.id_to_trainid = {-1: 255, 0: 255, 1: 0, 2: 1,
                              3: 2, 4: 3, 5: 4, 6: 5,
                              7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,
                              14: 13, 15: 14, 16: 15, 17: 16,
                              18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26,
                              28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36}

        self.class_weights = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
                   0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
                   2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
                   0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
                   1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
                   4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
                   3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
                   0.750738, 4.040773]

        self.ignore_label = 255
        self.seg_dir_train = []
        self.seg_dir_test = []



        try:
            with open(img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            for i in range(len(self.img_dir_train)):
                self.img_dir_train[i]=self.img_dir_train[i].split(".")[0]+".npy"

            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            for i in range(len(self.depth_dir_train)):
                self.depth_dir_train[i]=self.depth_dir_train[i].split(".")[0]+".npy"

            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            for i in range(len(self.label_dir_train)):
                self.seg_dir_train[i].append(self.label_dir_train[i].split(".")[0]+"seg.npy")


            with open(img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            for i in range(len(self.img_dir_test)):
                self.img_dir_test[i]=self.img_dir_test[i].split(".")[0]+".npy"


            with open(depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            for i in range(len(self.depth_dir_test)):
                self.depth_dir_test[i]=self.depth_dir_test[i].split(".")[0]+".npy"

            with open(label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
            for i in range(len(self.label_dir_test)):
                self.seg_dir_test[i].append(self.label_dir_test[i].split(".")[0]+"seg.npy")

        except:
            if data_dir is None:
                data_dir = '/path/to/SUNRGB-D'
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

            local_file_dir = '/'.join(img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(label_dir_test_file, 'w') as f:
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
        image = np.load(img_dir[idx])
        _label = np.load(label_dir[idx])
        _label_copy = _label.copy()
        for k, v in self.id_to_trainid.items():
            _label_copy[_label == k] = v

        label = Image.fromarray(_label_copy.astype(np.uint8))
        image = Image.fromarray(image.astype(np.uint8),mode='RGB')


        depth = None
        seg = None

        if self.cfg.MULTI_MODAL:
            depth = np.load(depth_dir[idx])
            depth = Image.fromarray(depth.astype(np.uint8),mode='RGB')
            # seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
            seg = np.load(seg_dir[idx])
            seg = Image.fromarray(seg.astype(np.uint8),mode='RGB')
        elif 'depth' == self.cfg.TARGET_MODAL:
            # depth = Image.open(depth_dir[idx]).convert('RGB')
            depth = np.load(depth_dir[idx])
            depth = Image.fromarray(depth.astype(np.uint8),mode='RGB')
        elif 'seg' == self.cfg.TARGET_MODAL:
            # seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
            seg = np.load(seg_dir[idx])
            seg = Image.fromarray(seg.astype(np.uint8),mode='RGB')

        sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)

        if self.transform:
            sample = self.transform(sample)

        return sample


class CityScapes(torch.utils.data.Dataset):
    def __init__(self,cfg=None,transform=None,phase_train=True,data_dir=None):
        train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
        val_dirs = ["frankfurt/", "munster/", "lindau/"]
        test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
        self.id_to_trainid = {-1:255, 0: 255, 1:255, 2:255,3:255, 4:255, 5:255, 6:255, 7:0, 8:1,
                         9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255,
                         19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255,
                         30:255, 31:16, 32:17, 33:18}

        self.class_weights = None
       #  self.class_weights=[2.5965083,  6.7422495,  3.5350077,  9.866795 ,  9.691752 ,
       #  9.369563 , 10.289785 ,  9.954636 ,  4.308077 ,  9.491024 ,
       #  7.6707582,  9.395554 , 10.3475065,  6.3950195, 10.226835 ,
       # 10.241277 , 10.280692 , 10.396961 , 10.05563]
        self.class_weights=[ 0.25965083, 0.67422495, 0.35350077,0.9866795,0.9691752 ,
        0.9369563,1.0289785,0.9954636 ,  0.4308077 ,  0.9491024 ,0.76707582,0.9395554,
        1.03475065,  0.63950195, 1.0226835,1.0241277,1.0280692,1.0396961 , 1.005563]
        self.cfg=cfg
        self.ignore_label = 255
        self.transform=transform
        if phase_train:
            trainFlag="train"
            file_dir=train_dirs
        else:
            trainFlag="val"
            file_dir=val_dirs
        self.img_dir = data_dir + "/leftImg8bit/"+trainFlag+"/"
        self.label_dir = data_dir + "/gtFine/"+trainFlag+"/"
        self.examples = []
        # count=0
        for train_dir in file_dir:
            train_img_dir_path = self.img_dir + train_dir
            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                if 't.png' not in file_name:
                    continue
                img_id = file_name.split("_leftImg8bit.png")[0]
                img_path = train_img_dir_path + file_name
                label_img_path = self.label_dir +train_dir+img_id + "_gtFine_labelIds.png"
                example = {}
                seg_path_npy = label_img_path.split("_gtFine_labelIds.npy")[0]+"_seg.npy"

                example["img_path"] = img_path
                example["label_path"] = label_img_path
                example["seg_path"] = seg_path_npy
                self.examples.append(example)
        self.num_examples = len(self.examples)
        # if phase_train and self.class_weights==None:
        #     self.class_weights=self.getClassWeight(self.examples)
        #     print(self.class_weights)

    def __len__(self):
        return self.num_examples
    def __getitem__(self, index):
        example = self.examples[index]
        image_path = example["img_path"]
        label_path = example["label_path"]
        seg_path = example["seg_path"]


        # image = np.load(img_path)
        # label = np.load(label_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 ch
        # image = self.examples[index]["image"]
        # label = self.examples[index]["label"]


        label_copy=label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = np.asarray(label_copy)
        # image=cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


        seg=None

        if self.cfg.NO_TRANS==False:
            if 'seg' == self.cfg.TARGET_MODAL:
                seg = (color_label_np(label_copy, ignore=self.ignore_label).astype(np.uint8))
                seg=cv2.cvtColor(np.asarray(seg), cv2.COLOR_RGB2BGR)
                # seg=np.load(seg_path)

                # seg = Image.fromarray(seg.astype(np.uint8),mode='RGB')
                # seg = self.examples[index]["seg"]
            sample = {'image': image,'label': label, 'seg': seg}
        else:
            # print(image.size)
            sample = {'image': image,'label': label}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def getClassWeight(self,example):
        trainId_to_count = {}
        class_weights = []
        num_classes=19
        for trainId in range(num_classes):
            trainId_to_count[trainId] = 0
        for step, samples in enumerate(example):
            if step % 100 == 0:
                print (step)

            # label_img = cv2.imread(label_img_path, -1)
            label=np.load(samples["label_img_path"])
            label_copy=label.copy()
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            label_img = Image.fromarray(label_copy.astype(np.uint8))
            for trainId in range(num_classes):
                # count how many pixels in label_img which are of object class trainId:
                trainId_mask = np.equal(label_img, trainId)
                trainId_count = np.sum(trainId_mask)
                # add to the total count:
                trainId_to_count[trainId] += trainId_count
        total_count = sum(trainId_to_count.values())
        for trainId, count in trainId_to_count.items():
            trainId_prob = float(count)/float(total_count)
            trainId_weight = 1/np.log(1.02 + trainId_prob)
            class_weights.append(trainId_weight*0.1)
        return class_weights

class Resize(transforms.Resize):

    def __call__(self, sample):

        for key in sample.keys():
            if key == 'label':
                sample['label'] = cv2.resize(sample['label'], self.size[::-1], interpolation=cv2.INTER_NEAREST)
                continue
            sample[key] = cv2.resize(sample[key], self.size[::-1], interpolation=cv2.INTER_LINEAR)
        return sample

class RandomScale(object):

    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):

        temp_scale = random.uniform(self.scale_low, self.scale_high)

        temp_aspect_ratio = 1.0
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        sample['image'] = cv2.resize(sample['image'], None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        sample['label'] = cv2.resize(sample['label'], None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return sample

class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='rand', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, sample):
        h, w = sample['label'].shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            sample['image'] = cv2.copyMakeBorder(sample['image'], pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            sample['label'] = cv2.copyMakeBorder(sample['label'], pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)

        h, w = sample['label'].shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        sample['image'] = sample['image'][h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        sample['label'] = sample['label'][h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            for key in sample.keys():
                sample[key] = cv2.flip(sample[key], 1)
        return sample


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __init__(self, ms_targets=[]):
        self.ms_targets = ms_targets
    def __call__(self, sample):
        image=sample['image']
        label=sample['label']
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        sample['image']=image
        sample['label']=label
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
            for t, m, s in zip(sample['image'], self.mean, self.std):
                t.sub_(m).div_(s)
        return sample