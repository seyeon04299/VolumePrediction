import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pretrainedmodels
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import sys
import model.norm as module_norm


######################################################################
####### Generative Models - DCGAN
######################################################################

class GeneratorBlock(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=0,dilation=1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU):
        super(GeneratorBlock, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.bn1 = norm_layer(outplanes)
        self.actv1 = actv_layer(inplace=True)

    def forward(self, x):
        
        out = self.convTrans1(x)
        out = self.bn1(out)
        out = self.actv1(out)

        return out


class Generator_DCGAN(nn.Module):
    def __init__(self, nz = 100, ngf=64, G_kernel_size=4):
        super(Generator_DCGAN, self).__init__()
        '''
        nz              : Input random vector length
        ngf             : Size of output feature map
        G_kernel_size   : Kernel size of Generator model
        '''
        
        kernel_size = G_kernel_size

        # stem
        self.conv0 = nn.ConvTranspose2d(nz, ngf*8, kernel_size=kernel_size, stride=1, padding=0, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.bn0 = nn.BatchNorm2d(ngf*8)
        self.actv0 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

        
        # Layers
        self.layer1 = self._make_layer(1, GeneratorBlock, ngf*8, kernel_size=3, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU)                   #out = 1024
        self.layer2 = self._make_layer(2, GeneratorBlock, ngf*4, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU)          #out = 512
        self.layer3 = self._make_layer(3, GeneratorBlock, ngf*4, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU)          #out = 256
        self.layer4 = self._make_layer(4, GeneratorBlock, ngf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU)          #out = 128
        self.layer5 = self._make_layer(5, GeneratorBlock, ngf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU)            #out = 64
        
        
        self.conv_final = nn.ConvTranspose2d(ngf, 3, kernel_size=kernel_size, stride=2, padding=1, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.tanh_final = nn.Tanh()


    def _make_layer(self, blk_id, block, inplanes, kernel_size=4, stride=1, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU):

        outplanes = int(inplanes/2) if blk_id >0 and blk_id%2==1 else int(inplanes)
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, padding=padding, dilation=1, norm_layer=norm_layer, actv_layer=actv_layer))

        return nn.Sequential(*layers)


    def forward(self, z):
        out = self.conv0(z)
        out = self.bn0(out)
        out = self.actv0(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)  
        out = self.layer4(out) 
        out = self.layer5(out) 
        
        out = self.conv_final(out)
        out = self.tanh_final(out)

        return out

####### Generator - HDCGAN

class Generator_HDCGAN(nn.Module):
    def __init__(self, nz = 100, ngf=64, G_kernel_size=4):
        super(Generator_HDCGAN, self).__init__()
        '''
        nz              : Input random vector length
        ngf             : Size of output feature map
        G_kernel_size   : Kernel size of Generator model
        '''
        
        kernel_size = G_kernel_size

        # stem
        self.conv0 = nn.ConvTranspose2d(nz, ngf*32, kernel_size=kernel_size, stride=1, padding=0, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.bn0 = nn.BatchNorm2d(ngf*32)
        self.actv0 = nn.SELU(inplace=True)
        
        # Layers
        self.layer1 = self._make_layer(1, GeneratorBlock, ngf*32, kernel_size=3, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)                   #out = 1024
        self.layer2 = self._make_layer(2, GeneratorBlock, ngf*16, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)          #out = 512
        self.layer3 = self._make_layer(3, GeneratorBlock, ngf*8, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)          #out = 256
        self.layer4 = self._make_layer(4, GeneratorBlock, ngf*4, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)          #out = 128
        self.layer5 = self._make_layer(5, GeneratorBlock, ngf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)            #out = 64
        
        
        self.conv_final = nn.ConvTranspose2d(ngf, 3, kernel_size=kernel_size, stride=2, padding=1, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.tanh_final = nn.Tanh()


    def _make_layer(self, blk_id, block, inplanes, kernel_size=4, stride=1, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU):
        outplanes = int(inplanes/2) if blk_id >0 else int(inplanes)
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, padding=padding, dilation=1, norm_layer=norm_layer, actv_layer=actv_layer))
        return nn.Sequential(*layers)


    def forward(self, z):
        out = self.conv0(z)
        out = self.bn0(out)
        out = self.actv0(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)  
        out = self.layer4(out) 
        out = self.layer5(out) 
        
        out = self.conv_final(out)
        out = self.tanh_final(out)

        return out



### DISCRIMINATOR ##########################################################################################################################

class DiscriminatorBlock(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(DiscriminatorBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = norm_layer(outplanes)
        self.actv1 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actv1(out)

        return out



class Discriminator_DCGAN(nn.Module):
    def __init__(self,ndf=64,D_kernel_size=4):
        '''
        ndf : size of feature map in Discriminator
        D_kernel_size : size of kernel (otherwise stated)
        '''
        super(Discriminator_DCGAN, self).__init__()
        kernel_size = D_kernel_size
        #stem
        self.conv0 = nn.Conv2d(3, ndf, kernel_size=kernel_size, stride=2, padding=1, bias=False)
        # self.bn0 = nn.BatchNorm2d(ndf)
        self.LeakyRelu0 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        

        self.layer1 = self._make_layer(1, DiscriminatorBlock, ndf, kernel_size=kernel_size, stride=2, padding=1, norm_layer=nn.BatchNorm2d)      # out 128
        self.layer2 = self._make_layer(2, DiscriminatorBlock, ndf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d)  # out 256
        self.layer3 = self._make_layer(3, DiscriminatorBlock, ndf*4, kernel_size=6, stride=2, padding = 0, norm_layer=nn.BatchNorm2d)  # out 512
        self.layer4 = self._make_layer(4, DiscriminatorBlock, ndf*8, kernel_size=6, stride=2, padding = 0, norm_layer=nn.BatchNorm2d)  # out 1024
        # self.layer5 = self._make_layer(5, DiscriminatorBlock, ndf*16, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, pooling=False)  # out 2048
        self.conv_final = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        

    def _make_layer(self, blk_id, block, inplanes, kernel_size=4, stride=1, padding = 0, norm_layer=nn.BatchNorm2d):
        
        outplanes = inplanes*2 if blk_id >0 else inplanes
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, padding=padding, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        # print('-------------------')
        # print('D_x : ',x.shape)         #D_x :  torch.Size([nc(3), 64, 64])
        out = self.conv0(x)
        # out = self.bn0(out)
        out = self.LeakyRelu0(out)
        # print('D_layer0',out.shape)     #D_layer0 torch.Size([ndf(64), 32, 32])
        out = self.layer1(out)
        # print('D_layer1',out.shape)     #D_layer1 torch.Size([128, 128, 54])
        out = self.layer2(out)
        # print('D_layer2',out.shape)     #D_layer2 torch.Size([256, 256, 27])
        out = self.layer3(out)
        # print('D_layer3',out.shape)     #D_layer3 torch.Size([512, 512, 13])
        out = self.layer4(out)
        # print('D_layer4',out.shape)     #D_layer3 torch.Size([512, 512, 13])
        # out = self.layer5(out)
        # print('D_layer5',out.shape)     #D_layer3 torch.Size([512, 512, 13])

        out = self.conv_final(out)
        out = self.sigmoid(out)
        # print('out shape:', out.shape)  
        # print('-------------------')
        
        return out




####### Discriminator - HDCGAN

class DiscriminatorBCBlock(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=1, norm_layer=nn.BatchNorm2d, actv_layer = nn.LeakyReLU):
        super(DiscriminatorBCBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = norm_layer(outplanes)
        self.actv1 = actv_layer(inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actv1(out)

        return out


class Discriminator_HDCGAN(nn.Module):
    def __init__(self,ndf=64,D_kernel_size=4):
        '''
        ndf : size of feature map in Discriminator
        D_kernel_size : size of kernel (otherwise stated)
        '''
        super(Discriminator_HDCGAN, self).__init__()
        kernel_size = D_kernel_size
        #stem
        self.conv0 = nn.Conv2d(3, ndf, kernel_size=kernel_size, stride=2, padding=1, bias=False)
        # self.bn0 = nn.BatchNorm2d(ndf)
        self.actv0 = nn.SELU(inplace=True)
        

        self.layer1 = self._make_layer(1, DiscriminatorBCBlock, ndf, kernel_size=kernel_size, stride=2, padding=1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)      # out 128
        self.layer2 = self._make_layer(2, DiscriminatorBCBlock, ndf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)  # out 256
        self.layer3 = self._make_layer(3, DiscriminatorBCBlock, ndf*4, kernel_size=6, stride=2, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)  # out 512
        self.layer4 = self._make_layer(4, DiscriminatorBCBlock, ndf*8, kernel_size=6, stride=2, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU)  # out 1024
        # self.layer5 = self._make_layer(5, DiscriminatorBlock, ndf*16, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=nn.BatchNorm2d, pooling=False)  # out 2048
        self.conv_final = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        

    def _make_layer(self, blk_id, block, inplanes, kernel_size=4, stride=1, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.SELU):
        
        outplanes = inplanes*2 if blk_id >0 else inplanes
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, padding=padding, norm_layer=norm_layer, actv_layer=actv_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        out = self.conv0(x)
        out = self.actv0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv_final(out)
        out = self.sigmoid(out)
        
        return out



##################################################################################################################################################################################################################
####### Generative Models - WGAN
##################################################################################################################################################################################################################

class GeneratorBlockWGAN(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=0, dilation=1, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU):
        super(GeneratorBlockWGAN, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.bn1 = norm_layer(outplanes)
        self.actv1 = actv_layer(inplace=True)

    def forward(self, x):
        
        out = self.convTrans1(x)
        out = self.bn1(out)
        out = self.actv1(out)

        return out

class GeneratorBlockWGAN_GP(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=0, dilation=1, norm_layer=nn.InstanceNorm2d, actv_layer=nn.ReLU):
        super(GeneratorBlockWGAN_GP, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.bn1 = norm_layer(outplanes, affine=True)
        self.actv1 = actv_layer(inplace=True)

    def forward(self, x):
        
        out = self.convTrans1(x)
        out = self.bn1(out)
        out = self.actv1(out)

        return out


class Generator_WGAN(nn.Module):
    def __init__(self, block, nz = 100, ngf=64, G_kernel_size=4,norm_layer='BatchNorm2d'):
        super(Generator_WGAN, self).__init__()
        '''
        nz              : Input random vector length
        ngf             : Size of output feature map
        G_kernel_size   : Kernel size of Generator model
        '''
        block = getattr(sys.modules[__name__], block)
        norm_layer = getattr(module_norm, norm_layer)
        kernel_size = G_kernel_size

        # stem
        self.conv0 = nn.ConvTranspose2d(nz, ngf*32, kernel_size=kernel_size, stride=1, padding=0, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.bn0 = norm_layer(ngf*32)
        self.actv0 = nn.ReLU(inplace=True)
        
        # Layers
        self.layer1 = self._make_layer(1, block, ngf*32, kernel_size=3, stride=2, padding = 1, norm_layer=norm_layer, actv_layer=nn.ReLU)                   #out = 1024
        self.layer2 = self._make_layer(2, block, ngf*16, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=norm_layer, actv_layer=nn.ReLU)          #out = 512
        self.layer3 = self._make_layer(3, block, ngf*8, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=norm_layer, actv_layer=nn.ReLU)          #out = 256
        self.layer4 = self._make_layer(4, block, ngf*4, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=norm_layer, actv_layer=nn.ReLU)          #out = 128
        self.layer5 = self._make_layer(5, block, ngf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=norm_layer, actv_layer=nn.ReLU)            #out = 64
        
        
        self.conv_final = nn.ConvTranspose2d(ngf, 3, kernel_size=kernel_size, stride=2, padding=1, output_padding=0, bias=False, dilation=1, padding_mode='zeros')
        self.tanh_final = nn.Tanh()


    def _make_layer(self, blk_id, block, inplanes, kernel_size=4, stride=1, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.ReLU):
        outplanes = int(inplanes/2) if blk_id >0 else int(inplanes)
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, padding=padding, dilation=1, norm_layer=norm_layer, actv_layer=actv_layer))
        return nn.Sequential(*layers)


    def forward(self, z):
        out = self.conv0(z)
        out = self.bn0(out)
        out = self.actv0(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)  
        out = self.layer4(out) 
        out = self.layer5(out) 
        
        out = self.conv_final(out)
        out = self.tanh_final(out)

        return out


class DiscriminatorBlockWGAN(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=1, norm_layer=nn.BatchNorm2d, actv_layer = nn.LeakyReLU):
        super(DiscriminatorBlockWGAN, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = norm_layer(outplanes)
        self.actv1 = actv_layer(0.2,inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.actv1(out)

        return out



class DiscriminatorBlockWGAN_GP(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=4, stride=1, padding=1, norm_layer=nn.InstanceNorm2d, actv_layer = nn.LeakyReLU):
        super(DiscriminatorBlockWGAN_GP, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = norm_layer(outplanes, affine=True)
        self.actv1 = actv_layer(0.2,inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.actv1(out)

        return out


class Discriminator_WGAN(nn.Module):
    def __init__(self,block, ndf=64,D_kernel_size=4,norm_layer='BatchNorm2d'):
        '''
        ndf : size of feature map in Discriminator
        D_kernel_size : size of kernel (otherwise stated)
        '''
        super(Discriminator_WGAN, self).__init__()
        block = getattr(sys.modules[__name__], block)
        norm_layer = getattr(module_norm, norm_layer)

        kernel_size = D_kernel_size
        #stem
        self.conv0 = nn.Conv2d(3, ndf, kernel_size=kernel_size, stride=2, padding=1, bias=False)
        # self.bn0 = nn.BatchNorm2d(ndf)
        self.actv0 = nn.LeakyReLU(0.2,inplace=True)
        

        self.layer1 = self._make_layer(1, block, ndf, kernel_size=kernel_size, stride=2, padding=1, norm_layer=norm_layer, actv_layer=nn.LeakyReLU)      # out 128
        self.layer2 = self._make_layer(2, block, ndf*2, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=norm_layer, actv_layer=nn.LeakyReLU)  # out 256
        self.layer3 = self._make_layer(3, block, ndf*4, kernel_size=6, stride=2, padding = 0, norm_layer=norm_layer, actv_layer=nn.LeakyReLU)  # out 512
        self.layer4 = self._make_layer(4, block, ndf*8, kernel_size=6, stride=2, padding = 0, norm_layer=norm_layer, actv_layer=nn.LeakyReLU)  # out 1024
        # self.layer5 = self._make_layer(5, DiscriminatorBlock, ndf*16, kernel_size=kernel_size, stride=2, padding = 1, norm_layer=norm_layer, pooling=False)  # out 2048
        self.conv_final = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=1, padding=0, bias=False)

        
    def _make_layer(self, blk_id, block, inplanes, kernel_size=4, stride=1, padding = 0, norm_layer=nn.BatchNorm2d, actv_layer=nn.LeakyReLU):
        
        outplanes = inplanes*2 if blk_id >0 else inplanes
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, padding=padding, norm_layer=norm_layer, actv_layer=actv_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        out = self.conv0(x)
        out = self.actv0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.conv_final(out)
        
        return out













##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################





######################################################################
####### Classification - CNN
######################################################################

class ResBlk_Basic(nn.Module):
    expansion = 1
    
    def __init__(self, blk_id, dropout, inplanes, outplanes, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, downsample=None, pooling=False):
        super(ResBlk_Basic, self).__init__()
        self.dropout = dropout
        padding = kernel_size//2+1 if (kernel_size//2)%2==0 else kernel_size//2
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = norm_layer(outplanes)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample if downsample is not None else None
        
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding) if pooling else None


    def forward(self, x):
        identity = x
        # print('identity = ', identity.shape)
        # print('X_shape = ', x.shape)
        out = self.conv1(x)
        # print('conv1 shape = ', out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print('conv2 shape = ', out.shape)
        out = self.bn2(out)
        # print(' out = ', out.shape)
        # print('downsample = ', self.downsample)
        if self.downsample is not None:
            identity = self.downsample(x)
        # print('identity2 = ', identity.shape)
        out += identity
        out = self.relu(out)

        out = self.dropout(out)
        # print(out.shape)
        
        # print('dropout out = ', out.shape)
        # if self.maxpool is not None:
        #     out = self.maxpool(out)

        return out



class ResBlk_Bottleneck(nn.Module):
    
    expansion=4
    
    def __init__(self, blk_id, dropout, inplanes, outplanes, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, downsample=None, pooling=False):
        super(ResBlk_Bottleneck, self).__init__()
        
        self.dropout = dropout
        padding = kernel_size//2+1 if (kernel_size//2)%2==0 else kernel_size//2
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = nn.Conv2d(outplanes,outplanes*self.expansion, kernel_size=1,stride=1,bias=False)
        self.bn3 = norm_layer(outplanes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample if downsample is not None else None
        
        # self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding) if pooling else None


    def forward(self, x):
        identity = x
        # print('identity = ', identity.shape)
        # print('X_shape = ', x.shape)
        out = self.conv1(x)
        # print('conv1 shape = ', out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        # print('conv2 shape = ', out.shape)
        out = self.bn2(out)
        # print(' out = ', out.shape)
        out = self.relu(out)
        
        out = self.conv3(out)
        # print('conv3 shape = ', out.shape)
        out = self.bn3(out)
        # print(' out = ', out.shape)
        # print('downsample = ', self.downsample)
        if self.downsample is not None:
            identity = self.downsample(x)
        # print('identity2 = ', identity.shape)
        out += identity
        out = self.relu(out)

        out = self.dropout(out)
        # print(out.shape)
        
        # print('dropout out = ', out.shape)
        # if self.maxpool is not None:
        #     out = self.maxpool(out)

        return out


class CustomResNet2d(nn.Module):

    def __init__(self, block, inplanes= 64, kernel_size=3, stride=2, num_classes=2, dropout=0.5, num_blocks_list=[3,4,6,3]):
        super(CustomResNet2d, self).__init__() 
        
        block = getattr(sys.modules[__name__], block)
        norm_layer=nn.BatchNorm2d
        self.dropout = dropout
        self.inplanes = inplanes
        
        # self._norm_layer = norm_layer    #nn.BatchNorm2d
        # self.inplanes = inplanes         #12
        self.dilation = 1
        # stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=stride, padding=7//2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=3//2)
        self.num_blocks_list = num_blocks_list


        self.layer1 = self._make_layer(0, block, dropout, inplanes, kernel_size, stride=1)
        self.layer2 = self._make_layer(1, block, dropout, inplanes, kernel_size, stride, norm_layer=norm_layer, pooling=True)
        self.layer3 = self._make_layer(2, block, dropout, inplanes*2, kernel_size, stride, norm_layer=norm_layer) #, pooling=True)
        self.layer4 = self._make_layer(3, block, dropout, inplanes*4, kernel_size, stride, norm_layer=norm_layer, pooling=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.fc = nn.Linear(inplanes*8, num_classes)
        self.fc = nn.Linear(inplanes*8*block.expansion, 4)
        self.fc2 = nn.Linear(1000, num_classes)
        
        ## site label --> fc layer
        
        # self.fc = nn.Linear(inplanes*8+3*224*224, num_classes)

    def _make_layer(self, blk_id, block, dropout, inplanes, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, residual=True, pooling=False):
        
        num_blocks = self.num_blocks_list[blk_id]
        outplanes = inplanes*2 if blk_id >0 else inplanes
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, outplanes*block.expansion, kernel_size=1, stride=stride, bias=False),
            norm_layer(outplanes*block.expansion)) if residual else None
        # print('-------------------------------------------')
        # print('block id', blk_id)
        # print('kernel_size = ', kernel_size)
        # print('self.inplanes = ', self.inplanes)
        # print('outplanes*block.expansion = ', outplanes*block.expansion)
        # print('stride = ', stride)
        # print('-------------------------------------------')
        
        
        layers = []
        layers.append(block(blk_id, dropout, self.inplanes, outplanes, kernel_size, stride, norm_layer=norm_layer, downsample=downsample, pooling=pooling))
        
        self.inplanes = outplanes*block.expansion
        
        for _ in range(1,num_blocks):
            layers.append(block(blk_id, dropout, self.inplanes, outplanes, kernel_size, stride=1, norm_layer=norm_layer, pooling=pooling))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        aux = torch.flatten(x,1)
        # print('x : ',x.shape, 'aux : ', aux.shape)
        x = self.conv1(x)
        # print('conv1 : ' ,x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('layer0',x.shape,'\n-------------------')
        x = self.layer1(x)
        # print('layer1',x.shape,'\n-------------------')
        x = self.layer2(x)
        # print('layer2',x.shape,'\n-------------------')
        x = self.layer3(x)
        # print('layer3',x.shape,'\n-------------------')
        x = self.layer4(x)
        # print('layer4',x.shape,'\n-------------------')
        
        x = self.avgpool(x)
        # print('avgpool',x.shape,'\n-------------------')
        x = torch.flatten(x, 1)
        # print('flatten',x.shape,'\n-------------------')
        ###
        # print('aux',aux.shape)
        # x = torch.cat((x,aux),1)
        # print('-------------------')
        # print('cat_x,aux',x.shape)
        x = self.fc(x)
        # x = self.fc2(x)
        # print('final',x.shape)
        return x




class CustomResNet2d_siteLabel(nn.Module):

    def __init__(self, block, inplanes= 64, kernel_size=3, stride=2, num_classes=2, dropout=0.5, num_blocks_list=[3,4,6,3],meta_inplanes=100):
        super(CustomResNet2d_siteLabel, self).__init__() 
        
        block = getattr(sys.modules[__name__], block)
        norm_layer=nn.BatchNorm2d
        self.dropout = dropout
        self.inplanes = inplanes
        
        # self._norm_layer = norm_layer    #nn.BatchNorm2d
        # self.inplanes = inplanes         #12
        self.dilation = 1
        # stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=stride, padding=7//2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=3//2)
        self.num_blocks_list = num_blocks_list


        self.layer1 = self._make_layer(0, block, dropout, inplanes, kernel_size, stride=1)
        self.layer2 = self._make_layer(1, block, dropout, inplanes, kernel_size, stride, norm_layer=norm_layer, pooling=True)
        self.layer3 = self._make_layer(2, block, dropout, inplanes*2, kernel_size, stride, norm_layer=norm_layer) #, pooling=True)
        self.layer4 = self._make_layer(3, block, dropout, inplanes*4, kernel_size, stride, norm_layer=norm_layer, pooling=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.fc = nn.Linear(inplanes*8, num_classes)
       

        self.meta = nn.Sequential(
            nn.Linear(4,meta_inplanes),
            nn.BatchNorm1d(meta_inplanes),
            nn.ReLU(),
            nn.Linear(meta_inplanes,int(meta_inplanes/2)),
            nn.BatchNorm1d(int(meta_inplanes/2)),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fc = nn.Linear(inplanes*8*block.expansion+int(meta_inplanes/2), 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, blk_id, block, dropout, inplanes, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, residual=True, pooling=False):
        
        num_blocks = self.num_blocks_list[blk_id]
        outplanes = inplanes*2 if blk_id >0 else inplanes
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, outplanes*block.expansion, kernel_size=1, stride=stride, bias=False),
            norm_layer(outplanes*block.expansion)) if residual else None
        
        
        layers = []
        layers.append(block(blk_id, dropout, self.inplanes, outplanes, kernel_size, stride, norm_layer=norm_layer, downsample=downsample, pooling=pooling))
        
        self.inplanes = outplanes*block.expansion
        
        for _ in range(1,num_blocks):
            layers.append(block(blk_id, dropout, self.inplanes, outplanes, kernel_size, stride=1, norm_layer=norm_layer, pooling=pooling))
        
        return nn.Sequential(*layers)


    def forward(self, x,site):
    
        aux = torch.flatten(x,1)
        # print('x : ',x.shape, 'aux : ', aux.shape)
        x = self.conv1(x)
        # print('conv1 : ' ,x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('layer0',x.shape,'\n-------------------')
        x = self.layer1(x)
        # print('layer1',x.shape,'\n-------------------')
        x = self.layer2(x)
        # print('layer2',x.shape,'\n-------------------')
        x = self.layer3(x)
        # print('layer3',x.shape,'\n-------------------')
        x = self.layer4(x)
        # print('layer4',x.shape,'\n-------------------')
        
        x = self.avgpool(x)
        # print('avgpool',x.shape,'\n-------------------')
        x = torch.flatten(x, 1)
        # print('flatten',x.shape,'\n-------------------')
        ###
        # print('aux',aux.shape)
        # x = torch.cat((x,aux),1)
        # print('-------------------')
        # print('cat_x,aux',x.shape)
        
        # print('fc',x.shape,'\n-------------------')
        # print(x)

        site = self.meta(site)
        x = torch.cat((x,site),1)
        
        x = self.fc(x)
        x = self.fc2(x)
        # print('final',x.shape)
        return x






class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


'''---------------Resnet--------------------- 

    관련 링크: https://pytorch.org/vision/stable/models.html
    Pretrain 시: normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 전처리 실행
                                     '''


def resnet18(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.resnet18(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return(model)


def resnet34(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.resnet18(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return(model)


def resnet50(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.resnet18(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return(model)


def resnet101(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.resnet18(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return(model)


def resnet152(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.resnet18(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return(model)




'''
class resnet18(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        
        if not pretrained:
            model = models.resnet18(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.resnet18(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            

        print(model)


        self = model


        

class resnet34(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.resnet34(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.resnet34(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model

class resnet50(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.resnet50(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.resnet50(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class resnet101(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.resnet101(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.resnet101(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model

class resnet152(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.resnet152(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.resnet152(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model
'''


'''---------------Inception Model--------------------- 

    관련 링크: https://pytorch.org/vision/stable/models.html
    Pretrain 시: normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 전처리 실행
                                     '''


def inception_v3(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.inception_v3(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.inception_v3(pretrained = pretrained)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    return(model)


def googlenet(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.googlenet(pretrained= pretrained, num_classes = num_classes)
    else:
        print("Warning , Aux layer not modified")
        model = models.googlenet(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    return(model)

'''

class inception_v3(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.inception_v3(pretrained= pretrained, num_classes = num_classes)
            print("Warning , Aux layer not modified")
        else:
            model = models.inception_v3(pretrained = pretrained)
            model.AuxLogits.fc = nn.Linear(768, num_classes)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        self = model

class googlenet(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.googlenet(pretrained= pretrained, num_classes = num_classes)
        else:
            print("Warning , Aux layer not modified")
            model = models.googlenet(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        self = model
'''

'''---------------Vggnet--------------------- 

    관련 링크: https://pytorch.org/vision/stable/models.html
    Pretrain 시: normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 전처리 실행
                                     '''


def vgg11(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg11(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg11(pretrained = pretrained)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)

def vgg11_bn(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg11_bn(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg11_bn(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)


def vgg13(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg13(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg13(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)

def vgg13_bn(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg13_bn(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg13_bn(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)

def vgg16(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg16(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg16(pretrained = pretrained)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)

def vgg16_bn(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg16_bn(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg16_bn(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)

def vgg19(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg19(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg19(pretrained = pretrained)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)

def vgg19_bn(num_classes =10, pretrained =False):  

    if not pretrained:
        model = models.vgg19_bn(pretrained= pretrained, num_classes = num_classes)
    else:
        model = models.vgg19_bn(pretrained = pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
                
    return(model)





'''    



class vgg11(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg11(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg11(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg11_bn(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg11_bn(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg11_bn(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg13(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg13(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg13(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg13_bn(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg13_bn(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg13_bn(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg16(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg16(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg16(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg16_bn(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg16_bn(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg16_bn(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg19(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg19(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg19(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model


class vgg19_bn(BaseModel):

    def __init__(self, num_classes =10 , pretrained =False):

        super().__init__()
        if not pretrained:
            model = models.vgg19_bn(pretrained= pretrained, num_classes = num_classes)
        else:
            model = models.vgg19_bn(pretrained = pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        self = model
'''

'''---------------Efficientnet--------------------- 

    관련 링크: 
    - https://pypi.org/project/efficientnet-pytorch/ 
    - https://arxiv.org/abs/1905.11946 
    
    Pretrain 시: normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 전처리 실행
                                     '''



def efficientnet_b0(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b0', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)
                
    return(model)

def efficientnet_b1(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b1', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b1',num_classes=num_classes)
                
    return(model)

def efficientnet_b2(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b2', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b2',num_classes=num_classes)
                
    return(model)

def efficientnet_b3(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b3', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=num_classes)
                
    return(model)

def efficientnet_b4(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b4', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=num_classes)
                
    return(model)

def efficientnet_b5(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b5', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=num_classes)
                
    return(model)

def efficientnet_b6(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b6', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b6',num_classes=num_classes)
                
    return(model)

def efficientnet_b7(num_classes =10, pretrained =False):  

    if not pretrained:
        model =EfficientNet.from_name('efficientnet-b7', num_classes = num_classes)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)
                
    return(model)



'''
class efficientnet_b0(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b0', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)
        self = model


class efficientnet_b1(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b1', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b1',num_classes=num_classes)
        self = model

class efficientnet_b2(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b2', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b2',num_classes=num_classes)
        self = model

class efficientnet_b3(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b3', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=num_classes)
        self = model

class efficientnet_b4(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b4', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=num_classes)
        self = model

class efficientnet_b5(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b5', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=num_classes)
        self = model

class efficientnet_b6(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b6', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b6',num_classes=num_classes)
        self = model

class efficientnet_b7(BaseModel):

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        if not pretrained:
            model =EfficientNet.from_name('efficientnet-b7', num_classes = num_classes)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)
        
        self = model


'''




