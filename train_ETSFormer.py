#!/bin/env python 
# -*- coding: utf-8 -*-
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric

from trainer.trainer_Autoformer import Trainer_Autoformer
from trainer.trainer_ETSFormer import Trainer_ETSFormer
import utils.optimizers as module_opt

import model.transformers.Autoformer as module_arch_Autoformer
import model.transformers.Reformer as module_arch_Reformer
import model.transformers.Informer as module_arch_Informer
import model.transformers.Transformer as module_arch_Transformer
import model.transformers.ETSFormer as module_arch_ETSFormer

from parse_config import ConfigParser_Autoformer

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
    print('data loader.shape : ',data_loader.subject_map.shape[0])
    # valid_data_loader = None
    valid_data_loader = data_loader.split_validation()
    print('valid data loader : ',valid_data_loader)

    
    print('prepare gpu training')
    device, device_ids = prepare_device(config['n_gpu'])

    # Build Model Architecture
    # model_G = config.init_obj('arch_G', module_arch_ETSFormer)
    
    # model_dict = {
    #         'Autoformer': module_arch_Autoformer,
    #         'Transformer': module_arch_Reformer,
    #         'Informer': module_arch_Informer,
    #         'Reformer': module_arch_Transformer,
    #     }
    model_dict = {
            'ETSFormer': module_arch_ETSFormer,
        }
    model_G = model_dict[model_name].Model(config).float()

    # model_D = config.init_obj('arch_D', module_arch_AST)

    logger.info(model_G)
    # logger.info(model_D)

    # prepare for (multi-device) GPU training
    model_G = model_G.to(device)
    # model_D = model_D.to(device)
    
    # Weight initialization
    # model.apply(initialize_weights)


    if len(device_ids) > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=device_ids)
        print("parallel GPU on ids ", device_ids)
        # model_D = torch.nn.DataParallel(model_D, device_ids=device_ids)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])

    
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    print('metrics : ',metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_G = filter(lambda p: p.requires_grad, model_G.parameters())
    # trainable_params_D = filter(lambda p: p.requires_grad, model_D.parameters())
    
    print('trainable_params : ',trainable_params_G)
    # print('trainable_params_D : ',trainable_params_D)

    ## CHECK OPTIMIZERS
    # optimizer_D = config.init_obj('optimizer_D', torch.optim, trainable_params_D)

    optimizer_G = config.init_obj('optimizer_G', torch.optim, trainable_params_G)
    

    

    # lr_scheduler_G = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_G)
    # lr_scheduler_D = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_D)

    # lr_scheduler = None

    criterion = torch.nn.MSELoss()
    # criterion = None
    trainer = Trainer_ETSFormer(model_G,
                        data_loader, 
                        valid_data_loader, 
                        criterion, 
                        metrics, 
                        config,
                        optimizer_G,
                        device=device)
    trainer.train_Autoformer()
    

if __name__ == '__main__':
    # ArgumentParser
    args = argparse.ArgumentParser(description='ASTransformer')
    
    
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r_G', '--resume_G', default=None, type=str,
                      help='path to latest checkpoint (GENERATOR) (default: None)')
    # args.add_argument('-r_D', '--resume_D', default=None, type=str,
    #                   help='path to latest checkpoint (DISCRIMINATOR) (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--MODEL', '--MODEL_TYPE'], type=str, target='arch_G;type'),
        CustomArgs(['--OPT_G', '--OPT_G_lr'], type=float, target='optimizer_G;args;lr'),
        CustomArgs(['--D_MOD', '--D_MODEL'], type=int, target='arch_G;args;d_model'),
        CustomArgs(['--E_LAYERS', '--E_LAYERS'], type=int, target='arch_G;args;e_layers'),
        CustomArgs(['--N_HEADS', '--N_HEADS'], type=int, target='arch_G;args;n_heads'),
        CustomArgs(['--SEQ_LEN', '--SEQ_LENGTH'], type=int, target='data_loader;args;seq_len'),
        CustomArgs(['--LABEL_LEN', '--LABEL_LENGTH'], type=int, target='data_loader;args;label_len'),
        CustomArgs(['--PRED_LEN', '--PREDICTION_LENGTH'], type=int, target='data_loader;args;pred_len'),
        CustomArgs(['--SEC', '--SECONDS'], type=int, target='data_loader;args;sec')
    ]
    config = ConfigParser_Autoformer.from_args(args, options)
    print('begin main')
    main(config)
