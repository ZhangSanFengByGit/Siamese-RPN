import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from IPython import embed
from .config import config


class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

#for feature propagation
        self.source_feature = None
        self.temple_norm = None

        self.kernel_pre = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 25, kernel_size=1, stride=1)
        )

        self.delta = [[-2,-2], [-1,-2], [0,-2], [1,-2], [2,-2], \
                 [-2,-1], [-1,-1], [0,-1], [1,-1], [2,-1], \
                 [-2,0], [-1,0], [0,0], [1,0], [2,0], \
                 [-2,1], [-1,1], [0,1], [1,1], [2,1], \
                 [-2,2], [-1,2], [0,2], [1,2], [2,2]]



    def propagate(self, kernels,b,c,h,w):
        prop_f = torch.empty([b,c,h,w], device='cuda')
        for bb in range(b):  
            for hh in range(h):   #24
                for ww in range(w):   #24
                    product = None
                    for order in range(5*5):   #25
                        if hh+self.delta[order][1]>=0 and hh+self.delta[order][1]<h \
                        and ww+self.delta[order][0]>=0 and ww+self.delta[order][0]<w:
                            cur = self.source_feature[bb,:,hh+self.delta[order][1], ww+self.delta[order][0]] * \
                                  kernels[ bb, hh*w+ww, int(order/5), int(order%5) ]

                            if product:
                                product = product + cur
                            else:
                                product = cur

                    prop_f[bb,:, hh, ww] = product

        #prop_f.register_hook(print)
        return prop_f


    def compute_weigt(self, embed_f):
        batch_size = embed_f.size(0)
        '''
        embed_norm = F.normalize(embed_f, p=2, dim=1) #problemantic: since it only L2 norm on the channel axis
        '''
        embed_norm = F.normalize(embed_f.view(batch_size, -1), p=2 ,dim=1)
        product = embed_norm * self.temple_norm
        weight = torch.sum(product, dim=1)
        '''
        weight = torch.mean(torch.sum(product, dim=1).view(batch_size, product.size(-1)* product.size(-1)), dim=1)
        '''
        assert weight.size()==torch.Size([batch_size]),'weight.size {}'.format(weight.size())
        return weight


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight.data, std=0.0005)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, std=0.0005)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight.data, std=0.0005)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, std=0.0005)


    #forward is used for training
    def forward(self, template, template_l, source, detection):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        template_l_feature = self.featureExtract(template_l)
        source_feature = self.featureExtract(source)
        detection_feature = self.featureExtract(detection)

        #prepare the temple
        self.source_feature = source_feature
        self.temple_norm = F.normalize(template_l_feature.view(N, -1), p=2 ,dim=1)
        #propagation phase
        b,c,h,w = detection_feature.size()
        concat_f = torch.cat((source_feature, detection_feature), dim=1)
        kernels = self.kernel_pre(concat_f)
        assert not torch.isnan(kernels).any()

        kernels = kernels.permute(0,2,3,1).contiguous().view(b, h*w, 5*5)
        kernels = F.softmax(kernels, dim=2).view(b, h*w, 5, 5)
        prop_feature = self.propagate(kernels,b,c,h,w)
        assert not torch.isnan(prop_feature).any()
        #compute the weights
        prop_w = self.compute_weigt(embed_prop)
        assert not torch.isnan(prop_w).any()
        x_w = self.compute_weigt(embed_x)
        assert not torch.isnan(x_w).any()

        weights = torch.cat( ( prop_w/(prop_w+x_w) , x_w/(prop_w+x_w) ), dim=0 )
        assert weights.size()==torch.Size([2, N]),'weights.size {}'.format(weights.size())
        assert not torch.isnan(weights[0]).any()
        assert not torch.isnan(weights[1]).any()
        #fuze the origin and the predicted
        prop_f_weighted = torch.empty(prop_feature.size(), device='cuda', requires_grad = False)
        detection_f_weighted = torch.empty(detection_feature.size(), device='cuda', requires_grad = False)
        for batch in range(N):
            prop_f_weighted[batch] = prop_feature[batch] * weights[0,batch]
            detection_f_weighted[batch] = detection_feature[batch] * weights[1,batch]
        assert prop_f_weighted.requires_grad == True
        assert detection_f_weighted.requires_grad == True

        fuzed_detection_f = prop_feature + detection_feature
        assert not torch.isnan(fuzed_detection_f).any()
        print('weights_0: {}'.format(weights[0]))
        print('weights_1: {}'.format(weights[1]))
        #formal correlation process
        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        conv_score = self.conv_cls2(fuzed_detection_f)
        conv_regression = self.conv_r2(fuzed_detection_f)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        return pred_score, pred_regression


    #test phase
    def track_init(self, template, template_l):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        template_l_feature = self.featureExtract(template_l)
        #init fot the whole sequence
        self.source_feature = None
        self.temple_norm = F.normalize(template_l_feature.view(N, -1), p=2 ,dim=1)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)



    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)

        #propagation part
        if self.source_feature:
            b,c,h,w = detection_feature.size()
            concat_f = torch.cat((source_feature, detection_feature), dim=1)
            kernels = self.kernel_pre(concat_f)
            kernels = kernels.permute(0,2,3,1).contiguous().view(b, h*w, 5*5)
            kernels = F.softmax(kernels, dim=2).view(b, h*w, 5, 5)
            prop_feature = self.propagate(kernels,b,c,h,w)
            #compute the weights
            prop_w = self.compute_weigt(embed_prop)
            x_w = self.compute_weigt(embed_x)
            weights = torch.cat( ( prop_w/(prop_w+x_w) , x_w/(prop_w+x_w) ), dim=0 )
            #fuze the origin and the predicted
            for batch in range(N):
                prop_feature[batch] = prop_feature[batch] * weights[0,batch]
                detection_feature[batch] = detection_feature[batch] * weights[1,batch]
            detection_feature = prop_feature + detection_feature
        self.source_feature = detection_feature
    
        #correlation process
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression
