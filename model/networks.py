import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .resnet import resnet18
from .resnet import resnet50
from .resnet import resnet101
import torchvision.models as models_tv
import model.resnet as models
import copy
import functools
import util.utils as util


batch_norm = nn.BatchNorm2d
models.BatchNorm = batch_norm
resnet18_place_path = '/data/dudapeng/pretrained_models/place/resnet18_places365.pth'


# resnet18_place_path = '/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth'

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_netowrks(cfg, device=None):
    # sync bn or not
    models.BatchNorm = batch_norm
    task_type = cfg.TASK_TYPE

    if task_type == 'segmentation':

        if cfg.MULTI_SCALE:
            # model = FCN_Conc_Multiscale(cfg, device=device)
            pass
        elif cfg.MULTI_MODAL:
            # model = FCN_Conc_MultiModalTarget_Conc(cfg, device=device)
            # model = FCN_Conc_MultiModalTarget_Late(cfg, device=device)
            model = FCN_Conc_MultiModalTarget(cfg, device=device)
        else:
            if cfg.MODEL == 'FCN':
                model = FCN_Conc(cfg, device=device)
            elif cfg.MODEL == 'trans2_seg':
                model = Trans2Seg(cfg, device=device)
            elif cfg.MODEL == 'FCN_MAXPOOL_FAKE':
                model = FCN_Conc_Maxpool_FAKE(cfg, device=device)
            # if cfg.MODEL == 'FCN_LAT':
            #     model = FCN_Conc_Lat(cfg, device=device)
            elif cfg.MODEL == 'UNET':
                model = UNet(cfg, device=device)
            # elif cfg.MODEL == 'UNET_256':
            #     model = UNet_Share_256(cfg, device=device)
            # elif cfg.MODEL == 'UNET_128':
            #     model = UNet_Share_128(cfg, device=device)
            # elif cfg.MODEL == 'UNET_64':
            #     model = UNet_Share_64(cfg, device=device)
            # elif cfg.MODEL == 'UNET_LONG':
            #     model = UNet_Long(cfg, device=device)
            elif cfg.MODEL == "PSP":
                model = PSPNet(cfg, device=device)
                # model = PSPNet(cfg, BatchNorm=nn.BatchNorm2d, device=device)

    elif task_type == 'recognition':
        if cfg.MODEL == 'trecg':
            model = TRecgNet_Scene_CLS(cfg, device=device)
        elif cfg.MODEL == 'trecg_compl':
            model = TrecgNet_Compl(cfg, device=device)
        elif cfg.MODEL == 'trecg_maxpool':
            model = TrecgNet_Scene_CLS_Maxpool(cfg, device=device)

    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]

def set_criterion(cfg, net):

    if 'CLS' in cfg.LOSS_TYPES or cfg.EVALUATE:
        criterion_cls = util.CrossEntropyLoss(weight=cfg.CLASS_WEIGHTS_TRAIN, ignore_index=cfg.IGNORE_LABEL)
        net.set_cls_criterion(criterion_cls)

    if 'SEMANTIC' in cfg.LOSS_TYPES:
        criterion_content = torch.nn.L1Loss()
        content_model = Content_Model(cfg, criterion_content)
        net.set_content_model(content_model)

    if 'PIX2PIX' in cfg.LOSS_TYPES:
        criterion_pix2pix = torch.nn.L1Loss()
        net.set_pix2pix_criterion(criterion_pix2pix)


def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)


##############################################################################
# Moduels
##############################################################################
class Conc_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True, upsample=True):
        super(Conc_Up_Residual, self).__init__()

        self.upsample = upsample
        if upsample:
            if dim_in == dim_out:
                kernel_size, padding = 3, 1
            else:
                kernel_size, padding = 1, 0

            self.smooth = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 3, 1

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_out, dim_out)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.smooth(x)
            residual = x

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.upsmaple:
            x += residual

        return self.relu(x)

class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out, norm=nn.BatchNorm2d):
        super(UpSample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0),
            # norm(dim_out),
            # nn.ReLU(True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            norm(dim_out),
            nn.ReLU(True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            norm(dim_out),
            nn.ReLU(True)
        )

    def forward(self, x, y=None):
        up_x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
        if y is not None:
            out = torch.cat([up_x, y], dim=1)
        else:
            out = up_x
        return self.model(out)


# class UpSample(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(UpSample, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
#             nn.InstanceNorm2d(dim_out)
#         )
#         self.relu = nn.ReLU(True)
#
#         self.conv2 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
#             nn.InstanceNorm2d(dim_out)
#         )
#
#     def forward(self, x, y=None):
#         up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         if y is not None:
#             out = torch.cat([up_x, y], dim=1)
#         else:
#             out = up_x
#         return self.relu(self.conv2(self.relu(self.conv1(out))))

class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True, upsample=True):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.upsample = upsample
        if dim_in == dim_out:
            kernel_size, padding = 3, 1
        else:
            kernel_size, padding = 1, 0

        self.smooth = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        else:
            dim_in = dim_out

        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)

        x = self.smooth(x)
        residual = x

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if self.upsample:
            x += residual

        return self.relu(x)


class Conc_Up_Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, upsample=None, norm=nn.BatchNorm2d, conc_feat=False):
        super(Conc_Up_Bottleneck, self).__init__()

        self.upsample = upsample
        if self.upsample:
            inplanes = planes

        dim_med = int(inplanes / 2)

        self.conv1 = nn.Conv2d(inplanes, dim_med, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(planes)

    def forward(self, x):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.upsample(x)
            residual = x
            x = self.relu(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Conc_Up_BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, upsample=None, norm=nn.BatchNorm2d, conc_feat=False):
        super(Conc_Up_BasicBlock, self).__init__()

        self.upsample = upsample

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.norm1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm(planes)

    def forward(self, x):

        residual = x

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.upsample(x)
            residual = x

        x = self.conv1(x)
        out = self.norm1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out) + residual

        return out


class Add_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, upsample=True):
        super(Add_Up_Residual_bottleneck, self).__init__()

        self.upsample = upsample
        if upsample:
            if dim_in == dim_out:
                kernel_size, padding = 3, 1
            else:
                kernel_size, padding = 1, 0

            self.smooth = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                norm(dim_out))

        dim_in = dim_out

        dim_med = int(dim_out / 2)

        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.smooth(x)
            residual = x

        if y is not None:
            x = x + y

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if self.upsample:
            x += residual

        return self.relu(x)


class Conc_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Residual_bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))
        # else:
        #     self.residual_conv = nn.Sequential(
        #         nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2,
        #                   padding=1, bias=False),
        #         norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        x = self.conv0(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Add_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Add_Residual_bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        dim_in = dim_out
        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):
        x = self.conv0(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = x + y

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


##############################################################################
# Trans2 Net
##############################################################################
class Content_Model(nn.Module):

    def __init__(self, cfg, criterion=None, in_channel=3):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.net = cfg.WHICH_CONTENT_NET

        if 'resnet' in self.net:
            from .pretrained_resnet import ResNet
            self.model = ResNet(self.net, cfg, in_channel=in_channel)

        fix_grad(self.model)
        # print_network(self)

    def forward(self, x, target, layers=None):

        # important when set content_model as the attr of trecg_net
        self.model.eval()

        layers = layers
        if layers is None or not layers:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        input_features = self.model((x + 1) / 2, layers)
        target_targets = self.model((target + 1) / 2, layers)
        len_layers = len(layers)
        loss_fns = [self.criterion] * len_layers
        alpha = [1] * len_layers

        content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
                          for i, gen_content in enumerate(input_features)]
        loss = sum(content_losses)
        return loss


class BaseTrans2Net(nn.Module):

    def __init__(self, cfg, device=None):
        super(BaseTrans2Net, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        self.arch = cfg.ARCH
        set_criterion(cfg, self)

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            self.pretrained = True
        else:
            self.pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__[self.arch](num_classes=365, deep_base=False)
            checkpoint = torch.load('./initmodel/' + self.arch + '_places365.pth', map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('load model pretrained on place')
        else:
            resnet = models.__dict__[self.arch](pretrained=self.pretrained, deep_base=False)

        self.maxpool = resnet.maxpool
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # if self.arch == 'resnet18':
        #     self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # else:
        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
        #                             resnet.conv3, resnet.bn3, resnet.relu)

        # self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(resnet.maxpool, resnet.layer1), resnet.layer2, resnet.layer3, resnet.layer4

        if self.trans:
            # self.layer1._modules['0'].conv1.stride = (2, 2)
            # if cfg.ARCH == 'resnet18':
            #     self.layer1._modules['0'].downsample = resnet.maxpool
            # else:
            #     self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)

            self.build_upsample_content_layers(dims)

        # self.init_type = 'kaiming'
        # if self.pretrained:
        #     if self.trans:
        #         init_weights(self.up1, self.init_type)
        #         init_weights(self.up2, self.init_type)
        #         init_weights(self.up3, self.init_type)
        #         init_weights(self.up4, self.init_type)
        #         init_weights(self.up5, self.init_type)
        # else:
        #     init_weights(self, self.init_type)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    # def _make_upsample(self, block, planes, blocks, stride=1, norm=nn.BatchNorm2d, conc_feat=False):
    #
    #     upsample = None
    #     if stride != 1:
    #         upsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1,
    #                       padding=0, bias=False),
    #             norm(planes)
    #         )
    #
    #     layers = []
    #
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, self.inplanes, norm=norm))
    #
    #     layers.append(block(self.inplanes, planes, upsample, norm, conc_feat))
    #
    #     self.inplanes = planes
    #
    #     return nn.Sequential(*layers)

    # def _make_upsample(self, block, planes, blocks, stride=1, norm=nn.BatchNorm2d, conc_feat=False):
    #
    #     upsample = None
    #     if stride != 1:
    #         if conc_feat:
    #             inplanes = self.inplanes * 2
    #         else:
    #             inplanes = self.inplanes
    #         upsample = nn.Sequential(
    #             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
    #                       padding=0, bias=False),
    #             norm(planes),
    #             nn.ReLU(True)
    #         )
    #         self.inplanes = planes
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, upsample, norm, conc_feat))
    #
    #     for i in range(1, blocks):
    #         layers.append(block(planes, planes, norm=norm))
    #
    #     self.inplanes = planes
    #
    #     return nn.Sequential(*layers)

    # def _make_agant_layer(self, inplanes, planes):
    #
    #     layers = nn.Sequential(
    #         nn.Conv2d(inplanes, planes, kernel_size=3,
    #                   stride=1, padding=1, bias=False),
    #         nn.InstanceNorm2d(planes),
    #         nn.ReLU(inplace=True)
    #     )
    #     return layers

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        if 'resnet18' == self.arch:
            inplanes = 512
            # self.up1 = self._make_upsample(Conc_Up_BasicBlock, dims[3], 5, stride=2, norm=norm)
            # self.up2 = self._make_upsample(Conc_Up_BasicBlock, dims[2], 4, stride=2, norm=norm, conc_feat=True)
            # self.up3 = self._make_upsample(Conc_Up_BasicBlock, dims[1], 3, stride=2, norm=norm, conc_feat=True)
            # self.up4 = self._make_upsample(Conc_Up_BasicBlock, dims[1], 3, stride=2, norm=norm, conc_feat=True)
            # self.up5 = self._make_upsample(Conc_Up_BasicBlock, dims[1], 2, stride=2, norm=norm, conc_feat=True)
            # self.up1 = UpSample(inplanes // 1 + 256, inplanes // 2)
            # self.up2 = UpSample(inplanes // 2 + 128, inplanes // 4)
            # self.up3 = UpSample(inplanes // 4 + 64, inplanes // 8)
            # self.up4 = UpSample(inplanes // 8 + 64, inplanes // 8)
            # self.up5 = UpSample(inplanes // 8, inplanes // 8)

            self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm)
            self.up5 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

            # self.skip1 = self._make_agant_layer(256, 256)
            # self.skip2 = self._make_agant_layer(128, 128)
            # self.skip3 = self._make_agant_layer(64, 64)
            # self.skip4 = self._make_agant_layer(64, 64)

        elif 'resnet50' in self.arch or 'resnet101' in self.arch:
            inplanes = 2048
            self.up1 = UpSample(inplanes // 1 + 1024, inplanes // 2)
            self.up2 = UpSample(inplanes // 2 + 512, inplanes // 4)
            self.up3 = UpSample(inplanes // 4 + 256, inplanes // 8)
            self.up4 = UpSample(inplanes // 8 + 128, inplanes // 16)
            self.up5 = UpSample(inplanes // 16, inplanes // 32)
            # self.up1 = self._make_upsample(Conc_Up_Bottleneck, dims[5], 5, stride=2, norm=norm)
            # self.up2 = self._make_upsample(Conc_Up_Bottleneck, dims[4], 4, stride=2, norm=norm, conc_feat=True)
            # self.up3 = self._make_upsample(Conc_Up_Bottleneck, dims[3], 3, stride=2, norm=norm, conc_feat=True)
            # self.up4 = self._make_upsample(Conc_Up_Bottleneck, dims[2], 2, stride=2, norm=norm, conc_feat=True)
            # self.up5 = self._make_upsample(Conc_Up_Bottleneck, dims[1], 2, stride=2, norm=norm, conc_feat=True)

        self.up_image = nn.Sequential(
            conv_norm_relu(dims[1], 64, norm=norm),
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        pass


class BaseTrans2Net_NoPooling(nn.Module):

    def __init__(self, cfg, device=None):
        super(BaseTrans2Net_NoPooling, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        self.arch = cfg.ARCH
        set_criterion(cfg, self)

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            self.pretrained = True
        else:
            self.pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__[self.arch](num_classes=365, deep_base=False)
            checkpoint = torch.load('./initmodel/' + self.arch + '_places365.pth', map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('content model pretrained using place')
        else:
            resnet = models.__dict__[self.arch](pretrained=self.pretrained, deep_base=False)

        self.maxpool = resnet.maxpool

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # else:
        #     self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
        #                                 resnet.conv3, resnet.bn3, resnet.relu)

        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        # self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(resnet.maxpool, resnet.layer1), resnet.layer2, resnet.layer3, resnet.layer4

        if self.trans:
            self.build_upsample_layers(dims)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if self.arch == 'resnet18':
            self.up1 = UpSample(dims[4] // 1 + 256, dims[4] // 2, norm=norm)
            self.up2 = UpSample(dims[4] // 2 + 128, dims[4] // 4, norm=norm)
            self.up3 = UpSample(dims[4] // 4 + 64, dims[4] // 8, norm=norm)
            self.up4 = UpSample(dims[4] // 8, dims[4] // 8, norm=norm)

            if self.cfg.MULTI_SCALE:
                dim_up_img = 128
            else:
                dim_up_img = 64

            # self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            # self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            # self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            # self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

        elif self.arch == 'resnet50':
            self.up1 = UpSample(dims[6] // 1 + 1024, dims[6] // 2, norm=norm)
            self.up2 = UpSample(dims[6] // 2 + 512, dims[6] // 4, norm=norm)
            self.up3 = UpSample(dims[6] // 4 + 256, dims[6] // 8, norm=norm)
            self.up4 = UpSample(dims[6] // 8, dims[6] // 16, norm=norm)
            # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
            # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
            # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm, conc_feat=False)

            if self.cfg.MULTI_SCALE:
                dim_up_img = 512
            else:
                dim_up_img = 128

        self.up_image = nn.Sequential(
            nn.Conv2d(dim_up_img, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        pass


class Trans2Seg(BaseTrans2Net):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.num_classes = cfg.NUM_CLASSES

        if 'resnet18' == cfg.ARCH:
            aux_dims = [256, 128, 64, 64]
            head_dim = 512
        elif 'resnet50' == cfg.ARCH:
            aux_dims = [1024, 512, 256, 128, 64]
            head_dim = 2048
        elif 'resnet101' == cfg.ARCH:
            aux_dims = [1024, 512, 256, 128, 64]
            head_dim = 2048

        self.score_head = _FCNHead(head_dim, self.num_classes, batch_norm)
        self.aux1 = self.define_aux_net(aux_dims[0] * 2)
        self.aux2 = self.define_aux_net(aux_dims[1] * 2)
        self.aux3 = self.define_aux_net(aux_dims[2] * 2)
        self.aux4 = self.define_aux_net(aux_dims[3] * 2, reduct=False)
        self.aux5 = self.define_aux_net(aux_dims[4], reduct=False)
        init_type = 'normal'

        if self.pretrained:

            # if self.trans:
            #     init_weights(self.up1, init_type)
            #     init_weights(self.up2, init_type)
            #     init_weights(self.up3, init_type)
            #     init_weights(self.up4, init_type)
            #     init_weights(self.up5, init_type)
            #     init_weights(self.up_image, init_type)

            for n, m in self.named_modules():
                if 'aux' in n or 'up' in n or 'score' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def define_aux_net(self, dim_in, reduct=True):

        if reduct:
            dim_out = int(dim_in / 4)
        else:
            dim_out = dim_in
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(dim_out, self.num_classes, kernel_size=1)
        )

    # def build_upsample_content_layers(self, dims):
    #
    #     norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
    #
    #     if 'resnet18' == self.cfg.ARCH:
    #         self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
    #         self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
    #
    #     elif 'resnet50' in self.cfg.ARCH or 'resnet101' in self.cfg.ARCH:
    #
    #         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
    #         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
    #         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
    #         # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
    #         # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
    #         # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         # self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        # self.up_image = nn.Sequential(
        #     nn.Conv2d(64, 3, 7, 1, 3, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # translation branch
            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3, layer_0)
            up5 = self.up5(up4)

            result['gen_img'] = self.up_image(up5)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            # segmentation branch
            score = self.score_head(layer_4)

            score_aux1 = self.aux1(torch.cat((layer_3, up1), 1))
            score_aux2 = self.aux2(torch.cat((layer_2, up2), 1))
            score_aux3 = self.aux3(torch.cat((layer_1, up3), 1))
            score_aux4 = self.aux4(torch.cat((layer_0, up4), 1))
            score_aux5 = self.aux5(up5)

            # if self.cfg.WHICH_SCORE == 'main' or not self.trans:
            #     score_aux1 = self.aux1(layer_3)
            #     score_aux2 = self.aux2(layer_2)
            #
            # elif self.cfg.WHICH_SCORE == 'up':
            #
            #     score_aux1 = self.aux1(up1)
            #     score_aux2 = self.aux2(up2)
            # elif self.cfg.WHICH_SCORE == 'both':
            #
            #     score_aux1 = self.aux1(layer_3) + self.aux1_up(up1)
            #     score_aux2 = self.aux2(layer_2) + self.aux2_up(up2)
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
            # score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux1
            # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux2
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux3
            score = F.interpolate(score, score_aux4.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux4
            score = F.interpolate(score, score_aux5.size()[2:], mode='bilinear', align_corners=True)
            result['cls'] = score + score_aux5


            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

            # if cal_loss:
            #     # aux1 = F.interpolate(score_aux1, size=source.size()[2:], mode='bilinear', align_corners=True)
            #     # aux2 = F.interpolate(score_aux2, size=source.size()[2:], mode='bilinear', align_corners=True)
            #     main_loss = self.cls_criterion(result['cls'], label)
            #     # aux1_loss = self.cls_criterion(aux1, label)
            #     # aux2_loss = self.cls_criterion(aux2, label)
            #     result['loss_cls'] = main_loss + 0.4 * aux1_loss + 0.1 * aux2_loss

        return result


class Trans2Seg_NoPooling(BaseTrans2Net_NoPooling):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.num_classes = cfg.NUM_CLASSES

        if 'resnet18' == cfg.ARCH:
            aux_dims = [256, 128, 64, 64]
            head_dim = 512
        elif 'resnet50' == cfg.ARCH:
            aux_dims = [1024, 512, 256, 128, 64]
            head_dim = 2048
        elif 'resnet101' == cfg.ARCH:
            aux_dims = [1024, 512, 256, 128, 64]
            head_dim = 2048

        self.score_head = _FCNHead(head_dim, self.num_classes, batch_norm)
        self.aux1 = self.define_aux_net(aux_dims[0] * 2)
        self.aux2 = self.define_aux_net(aux_dims[1] * 2)
        init_type = 'normal'

        if self.pretrained:

            # if self.trans:
            #     init_weights(self.up1, init_type)
            #     init_weights(self.up2, init_type)
            #     init_weights(self.up3, init_type)
            #     init_weights(self.up4, init_type)
            #     init_weights(self.up5, init_type)
            #     init_weights(self.up_image, init_type)

            for n, m in self.named_modules():
                if 'aux' in n or 'up' in n or 'score' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def define_aux_net(self, dim_in, reduct=True):

        if reduct:
            dim_out = int(dim_in / 4)
        else:
            dim_out = dim_in
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(dim_out, self.num_classes, kernel_size=1)
        )

    # def build_upsample_content_layers(self, dims):
    #
    #     norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
    #
    #     if 'resnet18' == self.cfg.ARCH:
    #         self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
    #         self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
    #
    #     elif 'resnet50' in self.cfg.ARCH or 'resnet101' in self.cfg.ARCH:
    #
    #         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
    #         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
    #         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
    #         # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
    #         # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
    #         # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         # self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        # self.up_image = nn.Sequential(
        #     nn.Conv2d(64, 3, 7, 1, 3, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # translation branch
            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            result['gen_img'] = self.up_image(up4)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            # segmentation branch
            score = self.score_head(layer_4)

            score_aux1 = self.aux1(torch.cat((layer_3, up1), 1))
            score_aux2 = self.aux2(torch.cat((layer_2, up2), 1))

            # if self.cfg.WHICH_SCORE == 'main' or not self.trans:
            #     score_aux1 = self.aux1(layer_3)
            #     score_aux2 = self.aux2(layer_2)
            #
            # elif self.cfg.WHICH_SCORE == 'up':
            #
            #     score_aux1 = self.aux1(up1)
            #     score_aux2 = self.aux2(up2)
            # elif self.cfg.WHICH_SCORE == 'both':
            #
            #     score_aux1 = self.aux1(layer_3) + self.aux1_up(up1)
            #     score_aux2 = self.aux2(layer_2) + self.aux2_up(up2)
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
            # score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux1
            # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux2
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

            # if cal_loss:
            #     # aux1 = F.interpolate(score_aux1, size=source.size()[2:], mode='bilinear', align_corners=True)
            #     # aux2 = F.interpolate(score_aux2, size=source.size()[2:], mode='bilinear', align_corners=True)
            #     main_loss = self.cls_criterion(result['cls'], label)
            #     # aux1_loss = self.cls_criterion(aux1, label)
            #     # aux2_loss = self.cls_criterion(aux2, label)
            #     result['loss_cls'] = main_loss + 0.4 * aux1_loss + 0.1 * aux2_loss

        return result


# class FCN_Conc_Maxpool_FAKE(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_Maxpool_FAKE, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#
#         self.source_net = FCN_Conc_Maxpool(cfg, device)
#         import copy
#         cfg_sample = copy.deepcopy(cfg)
#         cfg_sample.USE_FAKE_DATA = False
#         cfg_sample.NO_TRANS = True
#         self.compl_net = FCN_Conc_Maxpool(cfg_sample, device)
#
#     def set_content_model(self, content_model):
#         self.source_net.set_content_model(content_model)
#
#     def set_cls_criterion(self, criterion):
#         self.source_net.set_cls_criterion(criterion)
#         self.compl_net.set_cls_criterion(criterion)
#
#     def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
#
#         result_source = self.source_net(source, target, label, phase, content_layers, cal_loss=cal_loss)
#         input_compl = result_source['gen_img'].detach()
#         result_compl = self.compl_net(input_compl, None, label, phase, content_layers, cal_loss=cal_loss)
#
#         if phase == 'train':
#             result_source['loss_cls_compl'] = result_compl['loss_cls']
#         else:
#             result_source['cls'] += result_compl['cls']
#             # result_source['cls_compl'] = result_compl['cls']
#             # result_source['cls_fuse'] = (result_source['cls'] + result_compl['cls']) * 0.5
#
#         return result_source


class TrecgNet_Compl(nn.Module):

    def __init__(self, cfg, device=None):
        super(TrecgNet_Compl, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device

        self.source_net = TRecgNet_Scene_CLS(cfg, device)
        import copy
        cfg_sample = copy.deepcopy(cfg)
        cfg_sample.USE_FAKE_DATA = False
        cfg_sample.NO_TRANS = True
        cfg_sample.PRETRAINED = ''
        self.compl_net = TRecgNet_Scene_CLS(cfg_sample, device)
        for n, m in self.compl_net.layer4.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        self.compl_net.avgpool = nn.AvgPool2d(7, 1)

        set_criterion(cfg, self)
        set_criterion(cfg, self.source_net)
        set_criterion(cfg, self.compl_net)

        # self.fc_compl = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(512, cfg.NUM_CLASSES)
        # )

        # init_weights(self.compl_net, 'normal')
        # init_weights(self.fc_compl, 'normal')
        # self.avgpool = nn.AvgPool2d(14, 1)
        # for n, m in self.compl_net.layer4.named_modules():
        #     if 'conv1' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)

        # self.fc = nn.Sequential(
        #     conv_norm_relu(1024, 512, kernel_size=1, stride=1, padding=0),
        #     conv_norm_relu(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.AvgPool2d(14, 1),
        #     Flatten(),
        #     nn.Linear(512, self.cfg.NUM_CLASSES)
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, cfg.NUM_CLASSES)
        # )

    def set_content_model(self, content_model):
        self.source_net.set_content_model(content_model)

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.source_net.set_cls_criterion(criterion)
        self.compl_net.set_cls_criterion(criterion)
        self.cls_criterion = criterion

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):

        result_source = self.source_net(source, target, label, phase, content_layers, cal_loss=cal_loss)

        if self.cfg.INFERENCE:
            input_compl = target
        else:
            input_compl = result_source['gen_img'].detach()

        result_source['compl_source'] = input_compl
        result_compl = self.compl_net(input_compl, label=label)
        # cls_compl = self.fc_compl(F.avg_pool2d(feat_compl, feat_compl.size()[-1]))
        # result_compl = self.compl_net(input_compl, None, label, phase, content_layers, cal_loss=cal_loss)

        # conc_feat = torch.cat([result_source['avgpool'], result_compl['avgpool']], 1).to(self.device)
        # result_source['cls'] = self.fc(flatten(conc_feat))

        # cls_fuse = self.fc(cat)
        # if phase == 'train':
        #     result_source['loss_cls_compl'] = result_compl['loss_cls']
        #     result_source['loss_cls_fuse'] = self.cls_criterion(cls_fuse, label)
        # result_source['cls'] = cls_fuse
        if phase == 'train' and cal_loss:
            # result_source['loss_cls_compl'] = self.cls_criterion(cls_compl, label)
            result_source['loss_cls_compl'] = result_compl['loss_cls']
            result_source['loss_cls'] = self.cls_criterion(result_source['cls'], label)

        result_source['cls_compl'] = result_compl['cls']
        result_source['cls_original'] = result_source['cls']
        # result_source['avgpool_compl'] = result_compl['avgpool']
        alpha_main = 0.7
        result_source['cls'] = result_source['cls'] * alpha_main + result_source['cls_compl'] * (1-alpha_main)
        # result_source['cls'] = result_source['cls'] * 0.7 + result_source['cls_compl'] * 0.3

        return result_source


# class FCN_Conc_Maxpool(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_Maxpool, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
#
#         self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
#         # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
#         #                             resnet.conv3, resnet.bn3, resnet.relu)
#         self.maxpool = resnet.maxpool
#         self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
#
#         if self.trans:
#             if 'resnet50' in self.cfg.ARCH:
#                 for n, m in self.layer3.named_modules():
#                     if 'conv2' in n:
#                         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#                     elif 'downsample.0' in n:
#                         m.stride = (1, 1)
#                 for n, m in self.layer4.named_modules():
#                     if 'conv2' in n:
#                         m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#                     elif 'downsample.0' in n:
#                         m.stride = (1, 1)
#             # elif 'resnet18' in self.cfg.ARCH:
#             #     for n, m in self.layer4.named_modules():
#             #         if 'conv1' in n:
#             #             m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             #         elif 'downsample.0' in n:
#             #             m.stride = (1, 1)
#
#             self.build_upsample_content_layers(dims)
#
#         if 'resnet18' == cfg.ARCH:
#             aux_dims = [256, 128, 64]
#             head_dim = 512
#         elif 'resnet50' == cfg.ARCH:
#             aux_dims = [512, 256, 64]
#             head_dim = 2048
#
#         self.score_head = _FCNHead(head_dim, num_classes, batch_norm)
#
#         self.score_aux1 = nn.Sequential(
#             nn.Conv2d(aux_dims[0], num_classes, 1)
#         )
#
#         self.score_aux2 = nn.Sequential(
#             nn.Conv2d(aux_dims[1], num_classes, 1)
#         )
#         self.score_aux3 = nn.Sequential(
#             nn.Conv2d(aux_dims[2], num_classes, 1)
#         )
#
#         init_type = 'normal'
#         if pretrained:
#             init_weights(self.score_head, init_type)
#
#             if self.trans:
#                 init_weights(self.up1, init_type)
#                 init_weights(self.up2, init_type)
#                 init_weights(self.up3, init_type)
#                 init_weights(self.up4, init_type)
#                 init_weights(self.cross_layer_3, init_type)
#                 init_weights(self.cross_layer_4, init_type)
#
#             init_weights(self.score_head, init_type)
#             init_weights(self.score_aux3, init_type)
#             init_weights(self.score_aux2, init_type)
#             init_weights(self.score_aux1, init_type)
#
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_content_layers(self, dims):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         if 'resnet18' == self.cfg.ARCH:
#             self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
#             self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#             self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
#             self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         elif 'resnet50' in self.cfg.ARCH:
#             self.cross_layer_4 = nn.Conv2d(dims[6], dims[4], kernel_size=1, bias=False)
#             self.cross_layer_3 = nn.Conv2d(dims[5], dims[4], kernel_size=1, bias=False)
#
#             self.up1 = Conc_Residual_bottleneck(dims[5], dims[4], norm=norm)
#             self.up2 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#             self.up3 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm)
#             self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         self.up_image = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
#         result = {}
#
#         layer_0 = self.layer0(source)
#         layer_1 = self.layer1(self.maxpool(layer_0))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         if self.trans:
#             # translation branch
#             cross_layer4 = self.cross_layer_4(layer_4)
#             cross_layer3 = self.cross_layer_3(layer_3)
#
#             cross_conc = torch.cat((cross_layer4, cross_layer3), 1)
#
#             up1 = self.up1(cross_conc, layer_2)
#             up2 = self.up2(up1, layer_1)
#             up3 = self.up3(up2, layer_0)
#             up4 = self.up4(up3)
#
#             result['gen_img'] = self.up_image(up4)
#
#             if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
#                 result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:
#
#             # segmentation branch
#             score_head = self.score_head(layer_4)
#
#             score_aux1 = None
#             score_aux2 = None
#             score_aux3 = None
#             if self.cfg.WHICH_SCORE == 'main' or not self.trans:
#                 score_aux1 = self.score_aux1(layer_3)
#                 score_aux2 = self.score_aux2(layer_2)
#                 score_aux3 = self.score_aux3(layer_1)
#             elif self.cfg.WHICH_SCORE == 'up':
#                 score_aux1 = self.score_aux1(up1)
#                 # score_aux2 = self.score_aux2(up2)
#                 # score_aux3 = self.score_aux3(up3)
#
#             score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux1
#             score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux2
#             score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux3
#
#             result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
#
#             if cal_loss:
#                 result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


class FCN_Conc_MultiModalTarget(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_MultiModalTarget, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            checkpoint = torch.load(resnet18_place_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.score_head = _FCNHead(512, num_classes)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(dims[3] * 2, num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, num_classes, 1)
        )

        init_type = 'normal'
        if pretrained:

            init_weights(self.score_head, init_type)

            if self.trans:
                init_weights(self.up1_depth, init_type)
                init_weights(self.up2_depth, init_type)
                init_weights(self.up3_depth, init_type)
                init_weights(self.up4_depth, init_type)
                init_weights(self.up1_seg, init_type)
                init_weights(self.up2_seg, init_type)
                init_weights(self.up3_seg, init_type)
                init_weights(self.up4_seg, init_type)

            init_weights(self.score_aux3, init_type)
            init_weights(self.score_aux2, init_type)
            init_weights(self.score_aux1, init_type)
            init_weights(self.score_head, init_type)

        else:

            init_weights(self, init_type)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if 'bottleneck' in self.cfg.FILTERS:
            self.up1_depth = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2_depth = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3_depth = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4_depth = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

            self.up1_seg = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2_seg = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3_seg = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4_seg = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
        else:
            self.up1_depth = Conc_Up_Residual(dims[4], dims[3], norm=norm)
            self.up2_depth = Conc_Up_Residual(dims[3], dims[2], norm=norm)
            self.up3_depth = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_depth = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

            self.up1_seg = Conc_Up_Residual(dims[4], dims[3], norm=norm)
            self.up2_seg = Conc_Up_Residual(dims[3], dims[2], norm=norm)
            self.up3_seg = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_seg = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_depth = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_seg = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target_1=None, target_2=None, label=None, phase='train', content_layers=None,
                cal_loss=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch
            up1_depth = self.up1_depth(layer_4, layer_3)
            up2_depth = self.up2_depth(up1_depth, layer_2)
            up3_depth = self.up3_depth(up2_depth, layer_1)
            up4_depth = self.up4_depth(up3_depth)
            result['gen_depth'] = self.up_depth(up4_depth)

            up1_seg = self.up1_seg(layer_4, layer_3)
            up2_seg = self.up2_seg(up1_seg, layer_2)
            up3_seg = self.up3_seg(up2_seg, layer_1)
            up4_seg = self.up4_seg(up3_seg)
            result['gen_seg'] = self.up_seg(up4_seg)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
                result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            score_head = self.score_head(layer_4)

            score_aux1 = None
            score_aux2 = None
            score_aux3 = None
            if self.cfg.WHICH_SCORE == 'main':
                score_aux1 = self.score_aux1(layer_3)
                score_aux2 = self.score_aux2(layer_2)
                score_aux3 = self.score_aux3(layer_1)
            elif self.cfg.WHICH_SCORE == 'up':

                score_aux1 = self.score_aux1(torch.cat((up1_depth, up1_seg), 1))
                score_aux2 = self.score_aux2(torch.cat((up2_depth, up2_seg), 1))
                score_aux3 = self.score_aux3(torch.cat((up3_depth, up3_seg), 1))

            score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux3

            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        # if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
        #     result['loss_pix2pix_depth'] = self.pix2pix_criterion(result['gen_depth'], target_1)
        #     result['loss_pix2pix_seg'] = self.pix2pix_criterion(result['gen_seg'], target_2)

        return result


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=batch_norm):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


#######################################################################
class UNet(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            checkpoint = torch.load(resnet18_place_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        # norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=nn.BatchNorm2d)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=nn.BatchNorm2d)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=nn.BatchNorm2d)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=nn.BatchNorm2d, conc_feat=False)

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

        else:

            init_weights(self, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, label=None):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        result['cls'] = self.score(up4)
        result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):

    def __init__(self, cfg, bins=(1, 2, 3, 6), dropout=0.1,
                 zoom_factor=8, use_ppm=True, pretrained=True, device=None):
        super(PSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.BatchNorm = batch_norm
        self.device = device
        self.trans = not cfg.NO_TRANS
        self.cfg = cfg
        dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        if self.trans:
            self.build_upsample_content_layers(dims)

        resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=True)
        print("load ", cfg.ARCH)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.trans:
            self.layer1._modules['0'].conv1.stride = (2, 2)
            if cfg.ARCH == 'resnet18':
                self.layer1._modules['0'].downsample = resnet.maxpool
            else:
                self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)
            self.build_upsample_content_layers(dims)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, self.BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            self.BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, cfg.NUM_CLASSES, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                self.BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, cfg.NUM_CLASSES, kernel_size=1)
            )

        init_type = 'normal'
        if self.trans:
            init_weights(self.up0, init_type)
            init_weights(self.up1, init_type)
            init_weights(self.up2, init_type)
            init_weights(self.up3, init_type)
            init_weights(self.up4, init_type)
            init_weights(self.up5, init_type)
            init_weights(self.up_seg, init_type)
            init_weights(self.score_head, init_type)
            init_weights(self.score_aux1, init_type)
            init_weights(self.score_aux2, init_type)

        init_weights(self.aux, init_type)
        init_weights(self.cls, init_type)
        init_weights(self.ppm, init_type)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        # norm = self.norm
        # self.up0 = Conc_Up_Residual_bottleneck(dims[7], dims[6], norm=norm, upsample=False)

        self.cross_1 = nn.Conv2d(dims[6], dims[4], kernel_size=1, bias=False)
        self.cross_2 = nn.Conv2d(dims[5], dims[4], kernel_size=1, bias=False)

        self.up1 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm, upsample=False)
        self.up2 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
        # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
        # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
        # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        # self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        self.up_seg = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.score_aux1 = nn.Conv2d(1024, self.cfg.NUM_CLASSES, 1)
        self.score_aux2 = nn.Conv2d(512, self.cfg.NUM_CLASSES, 1)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_model(self, content_model):
        self.content_model = content_model.to(self.device)

    def forward(self, source, target=None, label=None, phase='train', content_layers=None, cal_loss=True, matrix=None):

        x = source
        y = label
        result = {}
        # x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        layer_0 = self.layer0(x)
        if not self.trans:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        x = layer_4
        if self.use_ppm:
            x = self.ppm(x)

        if not self.trans:
            x = self.cls(x)
            if self.zoom_factor != 1:
                result['cls'] = F.interpolate(x, size=source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                aux = self.aux(layer_3)
                if self.zoom_factor != 1:
                    aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                main_loss = self.cls_criterion(result['cls'], y)
                aux_loss = self.cls_criterion(aux, y)
                result['loss_cls'] = main_loss + 0.4 * aux_loss

        else:
            # up0_seg = self.up0(x, layer_4)
            # up1_seg = self.up1(up0_seg, layer_3)
            # up2_seg = self.up2(up1_seg, layer_2)
            # up3_seg = self.up3(up2_seg, layer_1)
            # up4_seg = self.up4(up3_seg, layer_0)
            # up5_seg = self.up5(up4_seg)
            # up1_seg = self.up1(layer_4, layer_3)
            # up2_seg = self.up2(up1_seg, layer_2)
            # up3_seg = self.up3(up2_seg, layer_1)
            # up4_seg = self.up4(up3_seg, layer_0)
            # up5_seg = self.up5(up4_seg)

            cross_1 = self.cross_1(layer_4)
            cross_2 = self.cross_2(layer_3)
            cross_conc = torch.cat((cross_1, cross_2), 1)
            up1_seg = self.up1(cross_conc, layer_2)
            up2_seg = self.up2(up1_seg, layer_1)
            up3_seg = self.up3(up2_seg, layer_0)
            up4_seg = self.up4(up3_seg)

            result['gen_img'] = self.up_seg(up4_seg)

            score_aux1 = self.score_aux1(cross_conc)
            score_aux2 = self.score_aux2(up1_seg)

            x = self.cls(x)
            score = F.interpolate(x, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1 + score_aux2
            # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux2
            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                # aux = self.aux(layer_3)
                # if self.zoom_factor != 1:
                #     aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                # main_loss = self.cls_criterion(result['cls'], y)
                # aux_loss = self.cls_criterion(aux, y)
                # result['loss_cls'] = main_loss + 0.4 * aux_loss
                result['loss_cls'] = self.cls_criterion(result['cls'], y)
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        return result


##################### Recognition ############################

class TRecgNet_Scene_CLS(BaseTrans2Net_NoPooling):

    def __init__(self, cfg, device=None):
        super(TRecgNet_Scene_CLS, self).__init__(cfg, device=device)

        self.avg_pool_size = 14
        self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = Evaluator(cfg, resnet.fc.in_features)

        if self.cfg.ARCH == 'resnet18':
            fc_input_nc = 512
        else:
            fc_input_nc = 2048

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(fc_input_nc, cfg.NUM_CLASSES)
        )

        init_type = 'normal'
        if self.cfg.PRETRAINED:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'skip' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        result['feat'] = layer_4
        result['avgpool'] = self.avgpool(layer_4)

        if 'CLS' in self.cfg.LOSS_TYPES:

            result['cls'] = self.fc(result['avgpool'])
            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        if self.trans:

            # skip1 = self.skip1(layer_3)
            # skip2 = self.skip2(layer_2)
            # skip3 = self.skip3(layer_1)

            # up1 = self.up1(layer_4)
            # up2 = self.up2(up1 + skip1)
            # up3 = self.up3(up2 + skip2)
            # up4 = self.up4(up3 + skip3)

            if not self.cfg.MULTI_SCALE:
                up1 = self.up1(layer_4, layer_3)
                up2 = self.up2(up1, layer_2)
                up3 = self.up3(up2, layer_1)
                up4 = self.up4(up3)
                result['gen_img'] = self.up_image(up4)
            else:
                up1 = self.up1(layer_4, layer_3)
                up2 = self.up2(up1, layer_2)
                result['gen_img'] = self.up_image(up2)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        return result


class TrecgNet_Scene_CLS_Maxpool(BaseTrans2Net):

    def __init__(self, cfg, device=None):
        super(TrecgNet_Scene_CLS_Maxpool, self).__init__(cfg, device)
        self.avgpool = nn.AvgPool2d(7, 1)
        if self.cfg.ARCH == 'resnet18':
            self.fc_input_nc = 512
        else:
            self.fc_input_nc = 2048
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, cfg.NUM_CLASSES)
        )

        init_type = 'normal'

        # self.aux1 = nn.Sequential(
        #     nn.AvgPool2d(14, 1),
        #     Flatten(),
        #     nn.Linear(fc_input_nc // 2, cfg.NUM_CLASSES)
        # )
        # self.aux2 = nn.Sequential(
        #     nn.AvgPool2d(28, 1),
        #     Flatten(),
        #     nn.Linear(fc_input_nc // 4, cfg.NUM_CLASSES)
        # )

        # for n, m in self.layer4.named_modules():
        #     if 'conv1' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)

        if self.pretrained:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'skip' in n or 'aux' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

    def set_sample_model(self, sample_model):
        self.sample_model = sample_model
        self.compl_net = sample_model.compl_net
        # import util.utils as util
        # cls_criterion = util.CrossEntropyLoss(weight=self.cfg.CLASS_WEIGHTS_TRAIN, device=self.device,
        #                                           ignore_index=self.cfg.IGNORE_LABEL)
        # self.compl_net.cls_criterion = cls_criterion.to(self.device)
        self.compl_net.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, self.cfg.NUM_CLASSES)
        )
        # self.fc_conc = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(self.fc_input_nc * 2, self.fc_input_nc),
        #     nn.ReLU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.fc_input_nc, self.cfg.NUM_CLASSES)
        # )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        result['feat'] = layer_4
        result['avgpool'] = self.avgpool(layer_4)

        if self.trans:
            # translation branch

            # skip1 = self.skip1(layer_3)
            # skip2 = self.skip2(layer_2)
            # skip3 = self.skip3(layer_1)
            # skip4 = self.skip4(layer_0)
            # up1 = self.up1(layer_4)
            # up2 = self.up2(up1 + skip1)
            # up3 = self.up3(up2 + skip2)
            # up4 = self.up4(up3 + skip3)
            # up5 = self.up5(up4 + skip4)
            # result['gen_img'] = self.up_image(up5)
            # up1 = self.up1(layer_4)
            # up2 = self.up2(torch.cat((up1, layer_3), 1))
            # up3 = self.up3(torch.cat((up2, layer_2), 1))
            # up4 = self.up4(torch.cat((up3, layer_1), 1))
            # up5 = self.up5(torch.cat((up4, layer_0), 1))

            # up1 = self.up1(layer_4, layer_3)
            # up2 = self.up2(up1, layer_2)
            # up3 = self.up3(up2, layer_1)
            # up4 = self.up4(up3, layer_0)
            # up5 = self.up5(up4)
            # result['gen_img'] = self.up_image(up5)

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            result['gen_img'] = self.up_image(up3)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES:

            result['cls'] = self.fc(result['avgpool'])

            if self.cfg.USE_COMPL_DATA:
                with torch.no_grad():
                    result_sample = self.sample_model(source, target, label, phase, content_layers, cal_loss=False)
                input_compl = result_sample['gen_img'].detach()
                result_compl = self.compl_net(input_compl, label=label, cal_loss=True)
                result['compl_source'] = input_compl

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)
                if self.cfg.USE_COMPL_DATA:
                    # conc = torch.cat([result['avgpool'], result_compl['avgpool']], 1)
                    # result['cls'] = self.fc_conc(conc)
                    # result['loss_cls'] = self.cls_criterion(result['cls'], label)
                    result['loss_cls_compl'] = result_compl['loss_cls']
                    # result['loss_cls_compl'] = result_compl['loss_cls'] + self.cls_criterion(result['cls'], label)

                    result['cls'] = result['cls'] * 0.7 + result_compl['cls'] * 0.3

        return result


class Fusion(nn.Module):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion, self).__init__()
        self.cfg = cfg
        self.device = device
        # self.rgb_model = rgb_model
        # self.depth_model = depth_model
        self.net_RGB = rgb_model
        self.net_depth = depth_model
        # self.net_RGB = self.construct_single_modal_net(rgb_model.source_net)
        # self.net_depth = self.construct_single_modal_net(depth_model.source_net)

        if cfg.FIX_GRAD:
            fix_grad(self.net_RGB)
            fix_grad(self.net_depth)

        self.avgpool = nn.AvgPool2d(7, 1)
        # self.fc = nn.Linear(1024 * 4, cfg.NUM_CLASSES)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(1024 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, cfg.NUM_CLASSES)
        )

        init_weights(self.fc, 'normal')

    # only keep the classification branch
    def construct_single_modal_net(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module

        ops = [model.conv1, model.bn1, model.relu, model.layer1, model.layer2,
               model.layer3, model.layer4]
        return nn.Sequential(*ops)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, input_rgb, input_depth, label, phase=None, cal_loss=True):

        result = {}
        rgb_specific = self.net_RGB(input_rgb, cal_loss=False)
        depth_specific = self.net_depth(input_depth, cal_loss=False)
        # self.smooth = conv_norm_relu(1024, 512)

        # rgb = self.avgpool(rgb_specific)
        # rgb = rgb.view(rgb.size(0), -1)
        # cls_rgb = self.rgb_model.fc(rgb)
        # out['cls'] = self.rgb_model.fc(x)

        # depth = self.avgpool(depth_specific)
        # depth = depth.view(depth.size(0), -1)
        # cls_depth = self.depth_model.fc(depth)
        # out['cls'] = self.depth_model.fc(x)

        # alpha = 0.6
        # out['cls'] = alpha * cls_rgb + (1 - alpha) * cls_depth

        cls_rgb = rgb_specific['cls']
        cls_depth = depth_specific['cls']

        # concat = torch.cat((rgb_specific['feat'].detach(), depth_specific['feat'].detach()), 1).to(self.device)
        # x = self.avgpool(concat)
        # result['cls'] = self.fc(x)

        alpha = 0.9
        result['cls'] = alpha * cls_rgb + (1-alpha) * cls_depth

        if cal_loss:

            # if 'SEMANTIC' in self.cfg.LOSS_TYPES and target is not None and phase == 'train':
            #     loss_content = self.content_model(out['gen_img'], target, layers=content_layers) * self.cfg.ALPHA_CONTENT

            if 'CLS' in self.cfg.LOSS_TYPES and not self.cfg.UNLABELED:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class Fusion_Trecg_Fake(nn.Module):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion_Trecg_Fake, self).__init__()
        self.cfg = cfg
        self.device = device
        self.rgb_model = rgb_model
        self.depth_model = depth_model

        fix_grad(self.rgb_model)
        fix_grad(self.depth_model)
        # fix_grad(self.rgb_model.compl_net)
        # fix_grad(self.depth_model.compl_net)

        # for n, m in self.rgb_model.named_modules():
        #     if 'up' in n:
        #         fix_grad(m)
        # for n, m in self.depth_model.named_modules():
        #     if 'up' in n:
        #         fix_grad(m)

        # self.net_RGB = self.construct_single_modal_net(rgb_model)
        # self.net_depth = self.construct_single_modal_net(depth_model)
        # self.rgb_net = self.rgb_model.net
        # self.rgb_compl_net = self.rgb_model.compl_net
        # self.depth_net = self.depth_model.net
        # self.depth_compl_net = self.depth_model.compl_net

        self.avgpool = nn.AvgPool2d(14, 1)

        # self.fc = nn.Linear(2048, cfg.NUM_CLASSES)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.cfg.NUM_CLASSES * 4, self.cfg.NUM_CLASSES),
        # )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, cfg.NUM_CLASSES)
        )

        # self.conc_convs = nn.Sequential(
        #     conv_norm_relu(2048, 1024, kernel_size=1, padding=0),
        #     conv_norm_relu(1024, 512, kernel_size=3, padding=1)
        # )

        self.flatten = flatten

        init_weights(self.fc, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def forward(self, input_rgb, input_depth, label, phase=None, cal_loss=True):

        result = {}

        rgb_result = self.rgb_model(input_rgb, cal_loss=False)
        depth_result = self.depth_model(input_depth, cal_loss=False)

        result['gen_depth'] = rgb_result['gen_img']
        result['gen_rgb'] = depth_result['gen_img']
        result['gen_img'] = depth_result['gen_img']
        # rgb_cls = self.flatten(rgb_result['cls_original'])
        # rgb_cls_compl = self.flatten(rgb_result['cls_compl'])
        # depth_cls = self.flatten(depth_result['cls_original'])
        # depth_cls_compl = self.flatten(depth_result['cls_compl'])

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
            result['loss_content'] = self.content_model(rgb_result['gen_img'], input_depth)
            result['loss_content'] += self.content_model(depth_result['gen_img'], input_rgb)

        # cls = torch.cat([rgb_cls, rgb_cls_compl, depth_cls, depth_cls_compl], 1)
        rgb_feat = rgb_result['feat']
        rgb_feat_compl = rgb_result['feat']
        depth_feat = depth_result['feat']
        depth_feat_compl = depth_result['feat']
        feat_conc = torch.cat([rgb_feat, rgb_feat_compl, depth_feat, depth_feat_compl], 1)

        # feat = self.conc_convs(feat_conc)
        result['cls'] = self.fc(flatten(self.avgpool(feat_conc)))


        # rgb_cls = self.flatten(rgb_result['avgpool'])
        # rgb_cls_compl = self.flatten(rgb_result['avgpool_compl'])
        # depth_cls = self.flatten(depth_result['avgpool'])
        # depth_cls_compl = self.flatten(depth_result['avgpool_compl'])
        # cls = torch.cat([rgb_cls, rgb_cls_compl, depth_cls, depth_cls_compl], 1)
        # self.smooth = conv_norm_relu(1024, 512)
        # rgb = self.avgpool(rgb_specific)
        # rgb = rgb.view(rgb.size(0), -1)
        # cls_rgb = self.rgb_model.fc(rgb)11Kkk
        # out['cls'] = self.rgb_model.fc(x)

        # depth = self.avgpool(depth_specific)
        # depth = depth.view(depth.size(0), -1)
        # cls_depth = self.depth_model.fc(depth)
        # out['cls'] = self.depth_model.fc(x)

        # alpha = 0.6
        # out['cls'] = alpha * cls_rgb + (1 - alpha) * cls_depth

        # result['cls'] = (rgb_cls + rgb_cls_compl + depth_cls + depth_cls_compl) * 0.25
        # result['cls'] = rgb_cls * 0.4 + rgb_cls_compl * 0.1 + depth_cls * 0.5 + depth_cls_compl * 0.1
        # result['cls'] = self.fc(cls)
        if cal_loss:

            # if 'SEMANTIC' in self.cfg.LOSS_TYPES and target is not None and phase == 'train':
            #     loss_content = self.content_model(out['gen_img'], target, layers=content_layers) * self.cfg.ALPHA_CONTENT

            if 'CLS' in self.cfg.LOSS_TYPES and not self.cfg.UNLABELED:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


########################### INFOMAX ###############################
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Source_Model(nn.Module):

    def __init__(self, cfg, device=None):
        super(Source_Model, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        pretrained = cfg.PRETRAINED
        self.encoder = self.cfg.ARCH
        if pretrained == 'imagenet' or pretrained == 'place':
            is_pretrained = True
        else:
            is_pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__[cfg.ARCH](num_classes=365)
            checkpoint = torch.load(resnet18_place_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place {0} loaded....'.format(cfg.ARCH))
        else:
            resnet = models.__dict__[cfg.ARCH](pretrained=pretrained)
            print('{0} pretrained:{1}'.format(cfg.ARCH, str(pretrained)))

        self.maxpool = resnet.maxpool  # 1/4

        if self.encoder == 'resnet18':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                        resnet.conv3, resnet.bn3, resnet.relu)
        # self.layer0 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.avg_pool = nn.AvgPool2d(7, 1)
        self.evaluator = Evaluator(cfg, input_nc=resnet.fc.in_features)
        # self.fc_z = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(512 * 7 * 7, 128)
        # )

        if self.trans or self.cfg.FT or self.cfg.RESUME:
            self.layer1._modules['0'].conv1.stride = (2, 2)
            if cfg.ARCH == 'resnet18':
                self.layer1._modules['0'].downsample = resnet.maxpool
            else:
                self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)

        if self.trans:
            self.build_upsample_layers()

        init_type = 'kaiming'
        print(self.__class__.__name__, ' ', init_type)
        if not is_pretrained:
            init_weights(self, init_type)
        else:
            for k, v in self._modules.items():
                if 'up' in k:
                    init_weights(v, init_type)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def build_upsample_layers(self):
        dims = [32, 64, 128, 256, 512, 1024, 2048]
        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if self.cfg.ARCH == 'resnet18':
            self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm)
            self.up5 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
        elif 'resnet50' in self.cfg.ARCH or 'resnet101' in self.cfg.ARCH:

            self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        self.up_image = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, x, target=None, label=None, class_only=False):
        out = {}

        layer_0 = self.layer0(x)
        if not self.trans and not self.cfg.FT and not self.cfg.RESUME:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        out['feat_1'] = layer_1
        layer_2 = self.layer2(layer_1)
        out['feat_2'] = layer_2
        layer_3 = self.layer3(layer_2)
        out['feat_3'] = layer_3
        layer_4 = self.layer4(layer_3)
        out['feat_4'] = layer_4
        out['avg_rgb'] = self.avg_pool(layer_4)
        # out['z'] = self.fc_z(layer_4)

        if class_only:
            out['pred'] = self.evaluator(out['avg_rgb'])
            return out

        if label is not None:
            out['pred'] = self.evaluator(out['avg_rgb'])
            out['cls_loss'] = self.cls_criterion(out['pred'], label)

        # if class_only:
        #
        #     lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator(avg_rgb)
        #     out['pred'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        #     return out
        #
        # if label is not None:
        #     lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator(avg_rgb)
        #     out['pred'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        #     out['cls_loss'] = self.cls_criterion(lgt_glb_mlp_rgb, label) + self.cls_criterion(lgt_glb_lin_rgb, label)

        if self.trans:
            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3, layer_0)
            up5 = self.up5(up4)

            gen = self.up_image(up5)
            out['gen_cross'] = gen

        return out


class Cross_Model(nn.Module):

    def __init__(self, cfg, device=None):
        super(Cross_Model, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        relu = nn.ReLU(True)
        norm = nn.BatchNorm2d

        # resnet = models.__dict__[self.cfg.ARCH](pretrained=False, deep_base=True)
        # print('{0} pretrained:{1}'.format(self.cfg.ARCH, str(False)))
        #
        # self.maxpool = resnet.maxpool  # 1/4
        #
        # layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # # self.model = nn.Sequential(
        # #     layer0, resnet.layer1, resnet.layer2, resnet.layer3
        # # )

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            relu,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            relu,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            relu,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            relu
        )

        # self.model = nn.Sequential(
        #     layer0,
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     norm(128),
        #     relu,
        #     models.BasicBlock(128, 128),
        #     models.BasicBlock(128, 128),
        #     models.BasicBlock(128, 128),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     norm(256),
        #     relu
        # )

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64)
        # )
        #
        # self.conv2 = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128)
        # )
        #
        # self.conv3 = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(256)
        # )
        #
        # self.conv4 = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512)
        #     nn.ReLU(inplace=True)
        # )
        #
        # # self.conv5 = nn.Sequential(
        # #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        # #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # #     # nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        # #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # #     # nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        # #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # #     # nn.BatchNorm2d(512),
        # #     nn.ReLU(inplace=True))
        #
        # self.model = nn.Sequential(
        #     self.conv1, self.conv2, self.conv3, self.conv4
        # )

        self.fc_z = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 28 * 28, 128)
        )

        self.d_cross = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, padding=1),
            # norm(512),
            relu,
            nn.Conv2d(512, 512, kernel_size=1),
            # norm(512),
            relu,
            nn.Conv2d(512, 1, kernel_size=1)
        )

        # self.d_inner = GANDiscriminator(cfg, device)

        init_weights(self, 'kaiming')

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def forward(self, x, target):
        out = {}

        feat_gen = self.model(x)
        feat_target = self.model(target)
        feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)

        # z_gen = self.fc_z(feat_gen)
        # z_target = self.fc_z(feat_target)

        out['feat_gen'] = feat_gen
        out['feat_target'] = feat_target

        if 'CROSS' in self.cfg.LOSS_TYPES:

            pos = torch.cat((feat_gen, feat_target), 1)
            neg = torch.cat((feat_gen, feat_target_neg), dim=1)

            Ej = -F.softplus(-self.d_cross(pos)).mean()
            Em = F.softplus(self.d_cross(neg)).mean()
            out['cross_loss'] = (Em - Ej)

            self_pos = torch.cat((feat_gen, feat_gen), 1)
            feat_self_neg = torch.cat((feat_gen[1:], feat_gen[0].unsqueeze(0)), dim=0)
            self_neg = torch.cat((feat_gen, feat_self_neg), dim=1)

            # Ej_self = -F.softplus(-self.d_cross(self_pos)).mean()
            # Em_self = F.softplus(self.d_cross(self_neg)).mean()
            # out['cross_loss_self'] = (Em_self - Ej_self)


        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            out['pix2pix_loss'] = self.pix2pix_criterion(x, target)

        return out


def flatten(x):
    return x.reshape(x.size(0), -1)


class Evaluator(nn.Module):
    def __init__(self, cfg, input_nc=512):
        super(Evaluator, self).__init__()
        self.block_glb_mlp = MLPClassifier(input_nc, cfg.NUM_CLASSES, n_hidden=input_nc * 2, p=0.2)
        self.is_ft = cfg.FT
        # self.block_glb_lin = \
        #     MLPClassifier(512, self.n_classes, n_hidden=None, p=0.0)

    def forward(self, ftr_1):
        '''
        Input:
          ftr_1 : features at 1x1 layer
        Output:
          lgt_glb_mlp: class logits from global features
          lgt_glb_lin: class logits from global features
        '''
        # collect features to feed into classifiers
        # - always detach() -- send no grad into encoder!
        if not self.is_ft:
            h_top_cls = flatten(ftr_1).detach()
        else:
            h_top_cls = flatten(ftr_1)
        # h_top_cls = flatten(ftr_1)
        # compute predictions
        lgt_glb_mlp = self.block_glb_mlp(h_top_cls)
        # lgt_glb_lin = self.block_glb_lin(h_top_cls)
        return lgt_glb_mlp


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

        init_weights(self, 'kaiming')

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.c0 = nn.Conv2d(in_channel, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 10 * 10 + 128, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        relu = nn.ReLU(True)
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, 512, kernel_size=1),
            relu,
            nn.Conv2d(512, 512, kernel_size=1),
            relu,
            nn.Conv2d(512, 1, kernel_size=1)
        )

        init_weights(self, 'kaiming')

    def forward(self, x):
        return self.model(x)


class GANDiscriminator(nn.Module):
    # initializers
    def __init__(self, cfg, device=None):
        super(GANDiscriminator, self).__init__()
        self.cfg = cfg
        self.device = device
        norm = nn.BatchNorm2d
        self.d_downsample_num = 4

        distribute = [
            nn.Conv2d(512, 1024, kernel_size=4, stride=2),
            # norm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=2, stride=2),
            # norm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
        ]

        self.criterion = nn.BCELoss() if cfg.NO_LSGAN else nn.MSELoss()
        if self.cfg.NO_LSGAN:
            distribute.append(nn.Sigmoid())

        self.distribute = nn.Sequential(*distribute)

    def forward(self, x, target):
        # distribution
        pred = self.distribute(x)

        if target:
            label = 1
        else:
            label = 0

        dis_patch = torch.FloatTensor(pred.size()).fill_(label).to(self.device)
        loss = self.criterion(pred, dis_patch)

        return loss


class GANDiscriminator_Image(nn.Module):
    # initializers
    def __init__(self, cfg, device=None):
        super(GANDiscriminator_Image, self).__init__()
        self.cfg = cfg
        self.device = device
        norm = nn.BatchNorm2d
        self.d_downsample_num = 4
        relu = nn.LeakyReLU(0.2)

        distribute = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            norm(64),
            relu,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            relu,
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            norm(256),
            relu,
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            norm(512),
            relu,
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            norm(1024),
            relu,
            # nn.Conv2d(1024, 1024, kernel_size=2, stride=2),
            # norm(1024),
            # relu,
            nn.Conv2d(1024, 1, kernel_size=1),
        ]

        self.criterion = nn.BCELoss() if cfg.NO_LSGAN else nn.MSELoss()
        if self.cfg.NO_LSGAN:
            distribute.append(nn.Sigmoid())

        self.distribute = nn.Sequential(*distribute)
        init_weights(self, 'normal')

    def forward(self, x, target):
        # distribution
        pred = self.distribute(x)

        if target:
            label = 1
        else:
            label = 0

        dis_patch = torch.FloatTensor(pred.size()).fill_(label).to(self.device)
        loss = self.criterion(pred, dis_patch)

        return loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class PriorDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.l0 = nn.Linear(in_channel, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

        init_weights(self, 'normal')

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))



class BasicBlockWithoutNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockWithoutNorm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

        if inplanes == planes:
            kernel_size, padding = 3, 1
        else:
            kernel_size, padding = 1, 0

        self.downsample = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False)

        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

