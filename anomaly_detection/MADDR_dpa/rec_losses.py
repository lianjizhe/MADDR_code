import os
from collections import defaultdict, OrderedDict
from enum import Enum
import torch
from torch import nn
from torch.nn import functional as F
import pdb
from anomaly_detection.dpa.feature_extractor import PretrainedVGG19FeatureExtractor
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def forward(self, x, fake_z, dec, label, zs, zc_logit):
        assert len(x.shape) == 4
        logits = torch.cat([zs, zc_logit], dim=1)
        new_f_x = logits.view(fake_z.shape)
        y = dec(new_f_x)

        losses = ((x - y) * (x - y)).sum(3).sum(2).sum(1) / (x.size(1) * x.size(2) * x.size(3))
        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class L1Loss(nn.Module):
    def __init__(self, reduction='none'):
        super(L1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean', 'pixelwise']
        self._reduction = reduction
        self.fc1 = nn.Linear(10,2)
        self.fc2 = nn.Linear(10,2)
    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean', 'pixelwise']
        self._reduction = reduction

    def forward(self, x, fake_z, dec, label, zs, zc_logit):
        assert len(x.shape) == 4

        # 0-lits, 1-covidct, 2-covidxray
        # 1->0 2->1
        label1 = label
        label1[label1==1] = 0
        label1[label1==2] = 1

        # 1->1 2->1      
        label2 = label
        label2[label2==2] = 1

        feature_covid_liver = zs[:,:10]
        feature_covid_ct_xray = zs[:,10:] 

        my_f_x_1 = self.fc1(feature_covid_liver)
        my_f_x_2 = self.fc2(feature_covid_ct_xray)
        label1 = label1.to(dtype=torch.long)
        cls_loss_1 = self.my_loss(my_f_x_1,label1)

        label2 = label2.to(dtype=torch.long)
        cls_loss_2 = self.my_loss(my_f_x_2,label2)
        
        # compactness loss
        com_loss = torch.mean(self.compactness_loss(3,zc_logit))
        cls_loss = cls_loss_1 + cls_loss_2

        logits = torch.cat([zs, zc_logit], dim=1)
        new_f_x = logits.view(fake_z.shape)

        y = dec(new_f_x)

        losses = torch.abs(x - y)
        if self._reduction == 'pixelwise':
            return losses

        losses = losses.sum(3).sum(2).sum(1) / (losses.size(1) * losses.size(2) * losses.size(3))

        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)  + cls_loss + com_loss
            # return torch.mean(losses)  + com_loss
        else:
            return torch.sum(losses)
            
    def my_loss(self,x,label):
        # loss_function = nn.CrossEntropyLoss()
        loss_function = FocalLoss()
        return loss_function(x,label)

    def compactness_loss(self, num_class, output):
        n = output.shape[0]
        loss = 1/(num_class * n) * torch.sum(n**2/(n-1)**2 * torch.std(output,axis=1) **2, axis = 0)
        return loss

    def focal_mseloss(self,x,y,cls_feature,label):
        loss_function = WeightedFocalMSELoss()
        return loss_function(x,y,cls_feature,label)


class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 reduction='mean',
                 img_weight=0,
                 feature_weights=None,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False):
        super(PerceptualLoss, self).__init__()
        """
        We assume that input is normalized with 0.5 mean and 0.5 std
        """

        assert reduction in ['none', 'sum', 'mean', 'pixelwise']

        MEAN_VAR_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'vgg19_ILSVRC2012_object_detection_mean_var.pt')

        self.vgg19_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.vgg19_std = torch.Tensor([0.229, 0.224, 0.225])

        if use_feature_normalization:
            self.mean_var_dict = torch.load(MEAN_VAR_ROOT)
        else:
            self.mean_var_dict = defaultdict(
                lambda: (torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))
            )

        self.reduction = reduction
        self.use_L1_norm = use_L1_norm
        self.use_relative_error = use_relative_error

        self.model = PretrainedVGG19FeatureExtractor()
        self.dropout = nn.Dropout(0.1)
        # self.fc4 = nn.Linear(10,2)

        self.set_new_weights(img_weight, feature_weights)

    def set_reduction(self, reduction):
        self.reduction = reduction

    def forward(self, x, fake_z, dec, label, zs, zc_logit):

        # pixel-wise prediction is implemented only if loss is obtained from one layer of vgg
        if self.reduction == 'pixelwise':
            assert (len(self.feature_weights) + (self.img_weight != 0)) == 1

        layers = list(self.feature_weights.keys())
        weights = list(self.feature_weights.values())

        logits = torch.cat([zs, zc_logit], dim=1)
        new_f_x = logits.view(fake_z.shape)

        y = dec(new_f_x)

        x = self._preprocess(x) # 真实图片
        y = self._preprocess(y) # 重建图片

        f_x = self.model(x, layers)
        f_y = self.model(y, layers)
        
        # pdb.set_trace()
        loss = None

        if self.img_weight != 0:
            loss = self.img_weight * self._loss(x, y)

        for i in range(len(f_x)):

            # put mean, var on right device
            mean, var = self.mean_var_dict[layers[i]]
            mean, var = mean.to(f_x[i].device), var.to(f_x[i].device)
            self.mean_var_dict[layers[i]] = (mean, var)

            # compute loss
            norm_f_x_val = (f_x[i] - mean) / var
            norm_f_y_val = (f_y[i] - mean) / var

            # pdb.set_trace()
            cur_loss = self._loss(norm_f_x_val, norm_f_y_val)

            # pdb.set_trace()
            if loss is None:
                loss = weights[i] * cur_loss
            else:
                loss += weights[i] * cur_loss

        loss /= (self.img_weight + sum(weights))

        with open("feature_mseloss.txt","a") as fw:
                fw.write(str(loss.mean().item())+'\n')

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            # print(loss.mean(),cls_loss,com_loss)
            loss_sum = loss.mean() #+ cls_loss + com_loss
            return loss_sum
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'pixelwise':
            loss = loss.unsqueeze(1)
            scale_h = x.shape[2] / loss.shape[2]
            scale_w = x.shape[3] / loss.shape[3]
            loss = F.interpolate(loss, scale_factor=(scale_h, scale_w), mode='bilinear')
            return loss
        else:
            raise NotImplementedError('Not implemented reduction: {:s}'.format(self.reduction))

    def set_new_weights(self, img_weight=0, feature_weights=None):
        self.img_weight = img_weight
        if feature_weights is None:
            self.feature_weights = OrderedDict({})
        else:
            self.feature_weights = OrderedDict(feature_weights)

    def _preprocess(self, x):
        assert len(x.shape) == 4

        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)

        # denormalize
        vector = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).to(x.device)
        x = x * vector + vector

        # normalize
        x = (x - self.vgg19_mean.reshape(1, 3, 1, 1).to(x.device)) / self.vgg19_std.reshape(1, 3, 1, 1).to(x.device)
        return x

    def _loss(self, x, y):
        if self.use_L1_norm:
            norm = lambda z: torch.abs(z)
        else:
            norm = lambda z: z * z

        diff = (x - y)
        if not self.use_relative_error:
            loss = norm(diff)
        else:
            means = norm(x).mean(3).mean(2).mean(1)
            means = means.detach()
            loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1))
            
        # perform reduction
        if self.reduction == 'pixelwise':
            return loss.mean(1)
        else:
            return loss.mean(3).mean(2).mean(1)

    def my_loss(self,x,label):
        # loss_function = nn.CrossEntropyLoss()
        loss_function = FocalLoss()
        return loss_function(x,label)

    def compactness_loss(self, num_class, output):
        n = output.shape[0]
        loss = 1/(num_class * n) * torch.sum(n**2/(n-1)**2 * torch.std(output,axis=1) **2, axis = 0)
        return loss

    def focal_mseloss(self,x,y,cls_feature,label):
        loss_function = WeightedFocalMSELoss()
        return loss_function(x,y,cls_feature,label)


class FocalLoss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):

        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes 
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        
    def forward(self, preds, labels):
       
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))      
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class RelativePerceptualL1Loss(PerceptualLoss):
    def __init__(self, reduction='mean', img_weight=0, feature_weights=None):
        super().__init__(
            reduction=reduction,
            img_weight=img_weight,
            feature_weights=feature_weights,
            use_feature_normalization=True,
            use_L1_norm=True,
            use_relative_error=True,)


class ReconstructionLossType(Enum):
    perceptual = 'perceptual'
    relative_perceptual_L1 = 'relative_perceptual_L1'
    l1 = 'l1'
    l2 = 'l2'
    compose = 'compose'


RECONSTRUCTION_LOSSES = {
    ReconstructionLossType.perceptual: PerceptualLoss,
    ReconstructionLossType.relative_perceptual_L1: RelativePerceptualL1Loss,
    ReconstructionLossType.l1: L1Loss,
    ReconstructionLossType.l2: L2Loss
}
