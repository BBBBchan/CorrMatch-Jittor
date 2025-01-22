import model.backbone.resnet as resnet
from model.backbone.xception import xception

import jittor as jt
from jittor import nn, Module
import math
from jittor import einops 


class DeepLabV3Plus(Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()
        self.is_corr = True

        if 'resnet' in cfg['backbone']:
            self.backbone = \
                resnet.__dict__[cfg['backbone']](cfg['pretrain'], multi_grid=cfg['multi_grid'],
                                                 replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU())

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU())

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

        if self.is_corr:
            self.corr = Corr(nclass=cfg['nclass'])
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout2d(0.1),
            )
        self.drop = nn.Dropout2d(0.5)

    def execute(self, x, need_fp=False, use_corr=False):
        dict_return = {}
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]
    
        if need_fp:
            feats_decode = self._decode(jt.concat((c1, nn.Dropout2d(0.5)(c1)), dim=0), 
                                       jt.concat((c4, nn.Dropout2d(0.5)(c4)), dim=0))
            outs = self.classifier(feats_decode)
            outs = nn.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            if use_corr:
                proj_feats = self.proj(c4)
                corr_out_dict = self.corr(proj_feats, out)
                dict_return['corr_map'] = corr_out_dict['corr_map']
                corr_out = corr_out_dict['out']
                corr_out = nn.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp

            return dict_return

        feats_decode = self._decode(c1, c4)
        out = self.classifier(feats_decode)
        out = nn.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if use_corr:
            proj_feats = self.proj(c4)
            corr_out_dict = self.corr(proj_feats, out)
            dict_return['corr_map'] = corr_out_dict['corr_map']
            corr_out = corr_out_dict['out']
            corr_out = nn.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
        dict_return['out'] = out
        return dict_return

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = nn.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = jt.concat([c1, c4], dim=1)
        feature = self.fuse(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
    return block


class ASPPPooling(Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def execute(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return nn.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU())
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU())

    def execute(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = jt.concat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class Corr(Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def execute(self, feature_in, out):
        dict_return = {}
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        h_out, w_out = out.shape[2], out.shape[3]
        out = nn.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = nn.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        
        f1 = einops.rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = einops.rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = einops.rearrange(out, 'n c h w -> n c (h w)')
        corr_map = jt.matmul(f1.transpose(1, 2), f2) / jt.sqrt(jt.array(f1.shape[1]).float())

        
        corr_map = nn.softmax(corr_map, dim=-1)
        corr_map_sample = self.sample(corr_map.detach(), h_in, w_in)
        dict_return['corr_map'] = self.normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)
        dict_return['out'] = einops.rearrange(jt.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        return dict_return

    def sample(self, corr_map, h_in, w_in):
        index = jt.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape
        corr_map = einops.rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = nn.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = einops.rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        range_ = corr_map.max(dim=1, keepdims=True)[0] - corr_map.min(dim=1, keepdims=True)[0]
        temp_map = (- corr_map.min(dim=1, keepdims=True)[0] + corr_map) / range_
        corr_map = (temp_map > 0.5)
        norm_corr_map = einops.rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        return norm_corr_map



