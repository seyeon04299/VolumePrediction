#!/bin/env python 
# -*- coding: utf-8 -*-
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.model_TimeGAN as module_arch_TimeGAN
from parse_config import ConfigParser
from trainer import Trainer_TimeGAN
# from trainer import Trainer, Trainer_WGAN, Trainer_ProGAN, Trainer_UNCGAN, Trainer_UNCGAN_M, Trainer_TimeGAN
from utils import prepare_device, initialize_weights


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed()

def main(config):
    logger = config.get_logger('train')

    # Set Model Name
    model_name = config['trainer']['model_name']
    print('Model Name : ', model_name)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    print('data loader : ',data_loader)
    # valid_data_loader = data_loader.split_validation()
    # print('valid data loader : ',valid_data_loader)
    
    print('prepare gpu training')
    device, device_ids = prepare_device(config['n_gpu'])

    # Build Model Architecture
    model = config.init_obj('arch', module_arch_TimeGAN)

    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    
    # Weight initialization
    # model.apply(initialize_weights)


    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])

    
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    print('metrics : ',metrics)
    print('Model parameters')
    print(model.parameters())



    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    print('trainable_params : ',trainable_params)

    ## CHECK OPTIMIZERS
    # optimizer_G = config.init_obj('optimizer_G', torch.optim, trainable_params_G)
    # optimizer_D = config.init_obj('optimizer_D', torch.optim, trainable_params_D)
    

    # lr_scheduler_G = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_G)
    # lr_scheduler_D = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_D)

    # lr_scheduler = None

    # criterion = torch.nn.BCELoss()
    criterion = None
    trainer = Trainer_TimeGAN(model, criterion, metrics, config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=None)
    trainer.train_TimeGAN()
    

if __name__ == '__main__':
    # ArgumentParser
    args = argparse.ArgumentParser(description='TimeGAN')
    
    
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--opt_E', '--E_LR'], type=float, target='optimizer_e;args;lr'),
        CustomArgs(['--opt_R', '--R_LR'], type=float, target='optimizer_r;args;lr'),
        CustomArgs(['--opt_G', '--G_LR'], type=float, target='optimizer_g;args;lr'),
        CustomArgs(['--opt_D', '--D_LR'], type=float, target='optimizer_d;args;lr'),
        CustomArgs(['--opt_S', '--S_LR'], type=float, target='optimizer_s;args;lr'),
        CustomArgs(['--dis_thres', '--Disc_Thres'], type=float, target='trainer;dis_thresh'),
        CustomArgs(['--gamma', '--Gamma'], type=int, target='trainer;gamma'),
    ]
    config = ConfigParser.from_args(args, options)
    print('begin main')
    main(config)
