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
import utils.optimizers as module_opt

import model.model_TimeGAN as module_arch_TimeGAN
import model.model_ASTransformer as module_arch_AST
from parse_config import ConfigParser_AST
from trainer import Trainer_TimeGAN
from trainer import Trainer_ASTransfomer
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
    valid_data_loader = None
    # valid_data_loader = data_loader.split_validation()
    # print('valid data loader : ',valid_data_loader)
    
    print('prepare gpu training')
    device, device_ids = prepare_device(config['n_gpu'])

    # Build Model Architecture
    model_G = config.init_obj('arch_G', module_arch_AST)
    model_D = config.init_obj('arch_D', module_arch_AST)

    logger.info(model_G)
    logger.info(model_D)

    # prepare for (multi-device) GPU training
    model_G = model_G.to(device)
    model_D = model_D.to(device)
    
    # Weight initialization
    # model.apply(initialize_weights)


    if len(device_ids) > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=device_ids)
        model_D = torch.nn.DataParallel(model_D, device_ids=device_ids)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])

    
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    print('metrics : ',metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_G = filter(lambda p: p.requires_grad, model_G.parameters())
    trainable_params_D = filter(lambda p: p.requires_grad, model_D.parameters())
    
    print('trainable_params : ',trainable_params_G)
    print('trainable_params_D : ',trainable_params_D)

    ## CHECK OPTIMIZERS
    optimizer_D = config.init_obj('optimizer_D', torch.optim, trainable_params_D)


    
    if config['optimizer_G']['type']=='OpenAIAdam':
        n_updates_total = (data_loader.dataset.__len__() // data_loader.batch_size)*config['trainer']['epochs']
        optimizer_G = module_opt.OpenAIAdam(
            trainable_params_G,
            lr = config['optimizer_G']['args']['lr'],
            schedule = config['optimizer_G']['args']['schedule'],
            warmup = config['optimizer_G']['args']['warmup'],
            t_total = n_updates_total,
            b1 = config['optimizer_G']['args']['b1'],
            b2 = config['optimizer_G']['args']['b2'],
            e = config['optimizer_G']['args']['e'],
            l2 = config['optimizer_G']['args']['l2'],
            vector_l2 = config['optimizer_G']['args']['vector_l2'],
            max_grad_norm = config['optimizer_G']['args']['max_grad_norm'],
        )
    else:
        optimizer_G = config.init_obj('optimizer_G', torch.optim, trainable_params_G)
    

    

    # lr_scheduler_G = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_G)
    # lr_scheduler_D = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_D)

    # lr_scheduler = None

    criterion = torch.nn.BCELoss()
    # criterion = None
    trainer = Trainer_ASTransfomer(model_G,
                        model_D, 
                        data_loader, 
                        valid_data_loader, 
                        criterion, 
                        metrics, 
                        config,
                        optimizer_G,
                        optimizer_D,
                        device=device)
    trainer.train_ASTransformer()
    

if __name__ == '__main__':
    # ArgumentParser
    args = argparse.ArgumentParser(description='ASTransformer')
    
    
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r_G', '--resume_G', default=None, type=str,
                      help='path to latest checkpoint (GENERATOR) (default: None)')
    args.add_argument('-r_D', '--resume_D', default=None, type=str,
                      help='path to latest checkpoint (DISCRIMINATOR) (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--OPT_D', '--OPT_D_lr'], type=float, target='optimizer_D;args;lr'),
        CustomArgs(['--OPT_G', '--OPT_G_lr'], type=float, target='optimizer_G;args;lr'),
        CustomArgs(['--D_MOD', '--D_MODEL'], type=int, target='arch_G;args;d_model'),
        CustomArgs(['--EMB_DIM', '--EMBEDDING_DIM'], type=int, target='arch_G;args;embedding_dim'),
        CustomArgs(['--DFF', '--d_ff'], type=int, target='arch_G;args;d_ff'),
        CustomArgs(['--LMBDA', '--lambda'], type=float, target='trainer;lmbda')
    ]
    config = ConfigParser_AST.from_args(args, options)
    print('begin main')
    main(config)
