import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
from scipy import linalg
import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt
from torch.autograd import Variable



def gev_loss(out_mean1, out_gamma1, out_sigma1, target):
    
    out_mean = out_mean1.clone()
    out_gamma = out_gamma1.clone()
    out_sigma = out_sigma1.clone()
    
    gamma_eps, sigma_eps = 1e-5, 1e-5
    out_gamma += gamma_eps
    out_sigma += sigma_eps 


    log_sigma = torch.log(out_sigma)
    # print('log_sigma = ', log_sigma)

    factor = torch.pow(out_sigma,-1)

    multiplier = (1+torch.pow(out_gamma,-1))
    resi = target-out_mean
    # print('resi = ', resi)
    resi_factor = resi*factor
    # print('resi_factor = ', resi_factor)
    resi_gamma = torch.abs(resi_factor*out_gamma+1)
    # print('resi_gamma = ', resi_gamma)
    log_resi = (torch.log(resi_gamma))*multiplier
    # print('log_resi = ', log_resi)

    resi_pow = torch.pow(resi_gamma,-out_gamma)
    # print('resi_pow = ', resi_pow)
    
    if torch.sum(log_sigma != log_sigma) > 0:
        print('log_sigma has nan')
        print(log_sigma.min(), log_sigma.max(), log_sigma.min(), log_sigma.max())
    if torch.sum(log_resi != log_resi) > 0:
        print('log_resi has nan')
    if torch.sum(resi_pow != resi_pow) > 0:
        print('resi_pow has nan')

    l = -log_sigma-log_resi-resi_pow
    l = torch.mean(l)
    
    return l




#     resi = torch.abs(out_mean - target)
# #     resi = (torch.log((resi*factor).clamp(min=1e-4, max=5))*out_beta).clamp(min=-1e-4, max=5)
#     resi = (resi*factor*out_beta).clamp(min=1e-6, max=50)
#     log_1alpha = torch.log(out_1alpha)
#     log_beta = torch.log(out_beta)
#     lgamma_beta = torch.lgamma(torch.pow(out_beta, -1))
    
#     if torch.sum(log_1alpha != log_1alpha) > 0:
#         print('log_1alpha has nan')
#         print(lgamma_beta.min(), lgamma_beta.max(), log_beta.min(), log_beta.max())
#     if torch.sum(lgamma_beta != lgamma_beta) > 0:
#         print('lgamma_beta has nan')
#     if torch.sum(log_beta != log_beta) > 0:
#         print('log_beta has nan')
    
#     l = resi - log_1alpha + lgamma_beta - log_beta
#     l = torch.mean(l)
    return l




#####################################################################
# Peak identification
#####################################################################
def peak_identification(y, lag, threshold, influence):
    '''
    Influence : determines the influence of signals on the algorithm's detection threshold
    '''
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = 0

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))



#####################################################################
# For ASTransformers
#####################################################################
def loss_quantile(mu:Variable, labels:Variable, quantile:Variable):
    loss = 0
    for i in range(mu.shape[1]):
        mu_e = mu[:, i]
        labels_e = labels[:, i]

        I = (labels_e >= mu_e).float()
        each_loss = 2*(torch.sum(quantile*(((labels_e -mu_e))*I)+ (1-quantile) *((mu_e- labels_e))*(1-I)))
        loss += each_loss

    return loss

def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu_en: (Variable) dimension [batch_size, context_len] - estimated mean at time step t
        sigma_en: (Variable) dimension [batch_size, context_len] - estimated standard deviation at time step t
        mu_en: (Variable) dimension [batch_size, predict_len] - estimated mean at time step t
        sigma_en: (Variable) dimension [batch_size, predict_len] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    loss = 0
    zero_index = (labels != 0)
    for i in range(mu.shape[1]):
        zero_index = (labels[:, i] != 0)
        mu_e = mu[:, i]
        sigma_e = sigma[:, i]
        labels_e = labels[:, i]
        distribution = torch.distributions.normal.Normal(mu_e, sigma_e)
        likelihood = distribution.log_prob(labels_e)
        each_loss = -torch.mean(likelihood)
        loss += each_loss
    return loss

# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]
    
def accuracy_MAPE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    #pred_samples = samples.shape[0]
    rou_pred = mu
    abs_diff = labels - rou_pred
    numerator += 2 * (torch.sum(rou * abs_diff[labels > rou_pred]) - torch.sum(
        (1 - rou) * abs_diff[labels <= rou_pred])).item()
    denominator += torch.sum(labels).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def quantile_loss(quantile:float, mu:torch.Tensor, labels:torch.Tensor):
    #gaussian = torch.distributions.normal.Normal(mu, sigma)
    #pred = gaussian.sample()
    I = (labels >= mu).float()
    diff = 2*(torch.sum(quantile*((labels-mu)*I)+ (1-quantile) *(mu-labels)*(1-I))).item()
    denom = torch.sum(torch.abs(labels)).item()
    q_loss = diff/denom
    return q_loss

def MAPE(mu:torch.Tensor, labels:torch.Tensor):
    zero_index = (labels != 0)
    diff = mu[zero_index] -labels[zero_index]
    lo = torch.mean(torch.abs(diff / labels[zero_index])) *100
    return lo

def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    mask = labels == 0
    mu[mask] = 0.

    abs_diff = np.abs(labels - mu)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < mu] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= mu] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)
    
    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result



###########################################################################################
# Plotting for tensorboard
##############################################3333#########################################
def plot_classes_preds_with_truth(predicted,truth):
        '''
        Generates matplotlib Figure using a trained network
        '''
        max_num = predicted.shape[0]
        
        if max_num>=10:
            
            # plot the images in the batch, along with predicted and true labels
            fig, ax = plt.subplots(nrows=10,ncols=1,figsize=(20, 15))
            x_ = np.arange(predicted.shape[1])
            for i in range(10):
            
                ax[i].plot(x_, predicted[i,:], x_, truth[i,:])

        else:
            
            fig, ax = plt.subplots(nrows=max_num,ncols=1,figsize=(20, 15))
            x_ = np.arange(predicted.shape[1])
            for i in range(max_num):
                ax[i].plot(x_, predicted[i,:], x_, truth[i,:])

        


        # for i in range(15):
        #     ax[i].plot(x_, sample[i,:,:])
        # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        # plt.plot(sample)

        return fig




def plot_classes_preds(sample):
        '''
        Generates matplotlib Figure using a trained network
        '''
        max_num = sample.shape[0]
        
        if max_num>=4:
            # plot the images in the batch, along with predicted and true labels
            fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(12, 6))
            x_ = np.arange(60)
            ax[0].plot(sample[0,:])
            ax[1].plot(sample[1,:])
            ax[2].plot(sample[2,:])
            ax[3].plot(sample[3,:])
            # for i in range(15):
            #     ax[i].plot(x_, sample[i,:,:])
            # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
            # plt.plot(sample)
        else:
            
            fig, ax = plt.subplots(nrows=max_num,ncols=1,figsize=(12, 6))
            # x_ = np.arange(predicted.shape[1])
            for i in range(max_num):
                ax[i].plot(sample[i,:])

        return fig




###########################################################################################
# Normalization
###########################################################################################

def normalize_max(x,p):
    '''
    Normalize with respect to max of the first p mins (ex. first 5 mins)
    '''
    max = np.amax(x[:p])
    x_normed = x/max
    return x_normed

def normalize_historic(x,mean,std):
    '''
    Normalize with respect to HISTORIC mean and std
    '''
    # means = np.ones((x.shape[0],))*mean
    # stds = np.ones((x.shape[0],))*std
    x_normed = np.true_divide(np.subtract(x,mean),std)
    return x_normed


def unnormalize_historic(x,mean,std):
    '''
    Unnormalize with respect to HISTORIC mean and std
    '''
    
    x_std = std.clone().detach().requires_grad_(True).squeeze(-1).repeat(x.shape[1],1).T*x
    x_unnormed = x_std+mean.clone().detach().requires_grad_(True).squeeze(-1).repeat(x_std.shape[1],1).T
    return x_unnormed

def normalize_historic_batch(x,mean,std):
    '''
    Normalize with respect to HISTORIC mean and std
    '''
    # means = np.ones((x.shape[0],))*mean
    # stds = np.ones((x.shape[0],))*std
    
    x_mean = x-mean.clone().detach().requires_grad_(True).squeeze(-1).repeat(x.shape[1],1).T
    x_normed = torch.divide(x_mean,std.clone().detach().requires_grad_(True).squeeze(-1).repeat(x_mean.shape[1],1).T)
    return x_normed


def unnormalize_then_normalize(x,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step):
    vols_unnormalized = unnormalize_historic(x[:,:,0],mean_prev,std_prev)
    multiplier = int(sec_step/prev_sec)
    # Merge volumes s.t intervals will be 'sec_step'
    vols_unnormalized = torch.sum(vols_unnormalized.reshape(vols_unnormalized.shape[0],int(vols_unnormalized.shape[1]/multiplier),-1),axis=-1)
    # Normalize with respect to historical mean of sec_step
    vol = normalize_historic_batch(vols_unnormalized,mean_step,std_step)
    data = vol.unsqueeze(-1)

    if x.shape[2]>1:
        #LOB
        lobs = x[:,:,1:]
        
        lobs = lobs[:,multiplier-1::multiplier,:]
        # print('vol.shape : ', vol.shape)
        # print('lobs.shape : ', lobs.shape)
        data = torch.concat((vol.unsqueeze(-1),lobs),axis=-1)
    
    return data.float() 





########################################################################
# Metrics For loss in UNCGAN
########################################################################




def bayeLq_loss(out_mean, out_log_var, target, q=2, k1=1, k2=1):
    var_eps = 1e-5
    out_var = var_eps + torch.exp(out_log_var)
    # out_log_var = torch.clamp(out_log_var, min=-3, max=3)
    # factor = torch.exp(-1*out_log_var) #no dropout grad_clipping b4 optim.step 
    factor = 1/out_var
    diffq = factor*torch.pow(torch.abs(out_mean-target), q)
#     diffq = torch.clamp(diffq, min=1e-5, max=1e3)
    
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(torch.log(out_var))
    
    loss = 0.5*(loss1 + loss2)
    return loss

def bayeGen_loss(out_mean, out_1alpha, out_beta, target):
    alpha_eps, beta_eps = 1e-5, 1e-1
    out_1alpha += alpha_eps
    out_beta += beta_eps 
    factor = out_1alpha
    resi = torch.abs(out_mean - target)
#     resi = (torch.log((resi*factor).clamp(min=1e-4, max=5))*out_beta).clamp(min=-1e-4, max=5)
    resi = (resi*factor*out_beta).clamp(min=1e-6, max=50)
    log_1alpha = torch.log(out_1alpha)
    log_beta = torch.log(out_beta)
    lgamma_beta = torch.lgamma(torch.pow(out_beta, -1))
    
    # if torch.sum(log_1alpha != log_1alpha) > 0:
    #     print('log_1alpha has nan')
    #     print(lgamma_beta.min(), lgamma_beta.max(), log_beta.min(), log_beta.max())
    # if torch.sum(lgamma_beta != lgamma_beta) > 0:
    #     print('lgamma_beta has nan')
    # if torch.sum(log_beta != log_beta) > 0:
    #     print('log_beta has nan')
    
    l = resi - log_1alpha + lgamma_beta - log_beta
    l = torch.mean(l)
    return l


def sigma_calc(alpha,beta):
    
    gammabeta1 = torch.exp(torch.lgamma(3*torch.pow(beta,-1)))
    gammabeta2 = torch.exp(torch.lgamma(3*torch.pow(beta,-1)))
    sigma = alpha*torch.sqrt(gammabeta1*torch.pow(gammabeta2,-1))
    return sigma

def bayeLq_loss1(out_mean, out_var, target, q=2, k1=1, k2=1):
    '''
    out_var has sigmoid applied to it and is between 0 and 1
    '''
    eps = 1e-7
    out_log_var = torch.log(out_var + eps)
    factor = 1/(out_var + eps)
#     print('im dbg2: ', factor.min(), factor.max())
    diffq = factor*torch.pow(out_mean-target, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(out_log_var)
#     print('im dbg: ', loss1.item(), loss2.item())
    loss = 0.5*(loss1 + loss2)
    return loss

def bayeLq_loss_n_ch(out_mean, out_log_var, target, q=2, k1=1, k2=1, n_ch=3):
    '''
    assumes uncertainty values are single channel
    '''
    out_log_var_nch = out_log_var.repeat(1,n_ch,1,1)

    factor = torch.exp(-out_log_var_nch)
    diffq = factor*torch.pow(out_mean-target, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(out_log_var) #does it have to be nch times?
    loss = 0.5*(loss1 + loss2)
    return loss

def Sinogram_loss(A, out_y, target, q=2):
    '''
    A = n_rows x (128x88)
    expected image: 128 x 88
    So load the variable, transpose it.
    incoming variable: out_y, target: n_batch x 1 x 88 x 128
    z = out_y.view(-1,n_batch) : (128x88) x n_batch
    Az = n_row x 1
    '''
    n_batch = out_y.shape[0]
    #sino = torch.mm(A, out_y.view(-1,n_batch))
    #na = 120, nb = 128;
    #sino = sino.view(na,nb)
    resi = torch.abs(torch.mm(A, out_y.view(-1,n_batch)) - torch.mm(A, target.view(-1,n_batch)))
#     print('sino dbg1: ', resi.min(), resi.max())
    resi = torch.pow(resi, q)
    return torch.mean(resi)

def bayeLq_Sino_loss(A, out_mean, out_log_var, target, q=2, k1=1, k2=1):
    n_batch = out_mean.shape[0]
    var_eps = 3e-3
    out_var = var_eps + torch.exp(out_log_var)
    
    resi = torch.abs(torch.mm(A, out_mean.view(-1,n_batch)) - torch.mm(A, target.view(-1,n_batch)))
#     x1 = torch.mm(A, out_mean.view(-1,n_batch)).view(-1).data.cpu().numpy()
#     x2 = torch.mm(A, target.view(-1,n_batch)).view(-1).data.cpu().numpy()
#     plt.subplot(1,2,1)
#     plt.hist(x1)
#     plt.subplot(1,2,2)
#     plt.hist(x2)
#     plt.show()
    sino_var_eps = 2e-2
    A_out_log_var = torch.log(torch.mm(A, out_var.view(-1,n_batch)) + sino_var_eps)
#     print(A_out_log_var)
    x1 = A_out_log_var.view(-1).data.cpu().numpy()
#      plt.subplot(1,2,1)
#     plt.hist(x1)
#     plt.show()
    factor = torch.exp(-1*A_out_log_var)
    
    diffq = factor*torch.pow(resi, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(A_out_log_var)
    
    loss = 0.5*(loss1 + loss2)
    return loss

def bayeLq_Sino_loss1(A, out_mean, out_var, target, q=2, k1=1, k2=1):
    eps = 1e-7
    n_batch = out_mean.shape[0]
    #print(A.shape, out_mean.shape, out_log_var.shape, target.shape)
    resi = torch.abs(torch.mm(A, out_mean.view(-1,n_batch)) - torch.mm(A, target.view(-1,n_batch)))
    resi = torch.clamp(resi, min=0, max=1e2)
    
    out_log_var = torch.log(out_var+eps)
    A_out_log_var = torch.log(torch.mm(A, out_var.view(-1,n_batch)))
    A_out_log_var = torch.clamp(A_out_log_var, min=-3, max=3)
    
    factor = torch.exp(-1*A_out_log_var)
    
    diffq = factor*torch.pow(resi, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(A_out_log_var)
    
    loss = 0.5*(loss1 + loss2)
    return loss





########################################################################
# Metrics For loss in UNCGAN
########################################################################



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value




def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma


def num_flat_features(x):
    size = x.size()[1:] # all dim except batch dim
    num_features = 1
    for s in size:
        num_features*=s
    return num_features

def gradient_penalty_ProGAN(model_D, real, fake, alpha, train_step, device):
    BATCH_SIZE, C, H, W = real.shape
    eta = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated = real*eta+fake*(1-eta)

    # calculate Critic (Discriminater) Scores
    prob_interpolated = model_D(interpolated, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=prob_interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty


def gradient_penalty(model_D, real, fake,device):
    BATCH_SIZE, T= real.shape
    
    eta = torch.rand((BATCH_SIZE,1)).repeat(1,T).to(device)
    interpolated = real*eta+fake*(1-eta)

    # calculate Critic (Discriminater) Scores
    prob_interpolated = model_D(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=prob_interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty



def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

def initialize_weights(net):
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            if np.isscalar(value):
                self.writer.add_scalar(key, value)
                self._data.total[key] += value * n
                self._data.counts[key] += n
                self._data.average[key] = self._data.total[key] / self._data.counts[key]
            else:
                self._data.total[key] += value 
                self._data.counts[key] += 1
                self._data.average[key] += value

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

'''
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
'''