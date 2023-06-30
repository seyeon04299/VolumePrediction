#!/bin/env python 
# -*- coding: utf-8 -*-
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric


from trainer.trainer_Proformer import Trainer_Proformer
import utils.optimizers as module_opt

import model.transformers.Autoformer as module_arch_Autoformer
import model.transformers.Reformer as module_arch_Reformer
import model.transformers.Informer as module_arch_Informer
import model.transformers.Transformer as module_arch_Transformer

from parse_config import ConfigParser_AutoProformer

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
    valid_data_loader = data_loader.split_validation()

    print('data loader : ',data_loader)
    print('data loader.shape : ',data_loader.subject_map.shape[0])
    

    
    print('prepare gpu training')
    device, device_ids = prepare_device(config['n_gpu'])

    # Build Model Architecture
    # model_G = config.init_obj('arch_G', module_arch_Autoformer)
    
    model_dict = {
            'Autoformer': module_arch_Autoformer,
            'Reformer': module_arch_Reformer,
            'Informer': module_arch_Informer,
            'Transformer': module_arch_Transformer,
            'AutoProformer':module_arch_Autoformer,
            'TransProformer': module_arch_Transformer,
        }
    model_G0 = model_dict[model_name].Proformer(config,step=0).float()
    model_G1 = model_dict[model_name].Proformer(config,step=1).float()
    model_G2 = model_dict[model_name].Proformer(config,step=2).float()
    model_G3 = model_dict[model_name].Proformer(config,step=3).float()
    model_G4 = model_dict[model_name].Proformer(config,step=4).float()
    model_G5 = model_dict[model_name].Proformer(config,step=5).float()
    model_G6 = model_dict[model_name].Proformer(config,step=6,final=True).float()
    

    logger.info(model_G0)
    logger.info(model_G1)
    logger.info(model_G2)
    logger.info(model_G3)
    logger.info(model_G4)
    logger.info(model_G5)
    logger.info(model_G6)

    # prepare for (multi-device) GPU training
    model_G0 = model_G0.to(device)
    model_G1 = model_G1.to(device)
    model_G2 = model_G2.to(device)
    model_G3 = model_G3.to(device)
    model_G4 = model_G4.to(device)
    model_G5 = model_G5.to(device)
    model_G6 = model_G6.to(device)
    

    if len(device_ids) > 1:
        model_G0 = torch.nn.DataParallel(model_G0, device_ids=device_ids)
        model_G1 = torch.nn.DataParallel(model_G1, device_ids=device_ids)
        model_G2 = torch.nn.DataParallel(model_G2, device_ids=device_ids)
        model_G3 = torch.nn.DataParallel(model_G3, device_ids=device_ids)
        model_G4 = torch.nn.DataParallel(model_G4, device_ids=device_ids)
        model_G5 = torch.nn.DataParallel(model_G5, device_ids=device_ids)
        model_G6 = torch.nn.DataParallel(model_G6, device_ids=device_ids)
        # model_D = torch.nn.DataParallel(model_D, device_ids=device_ids)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])

    
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    print('metrics : ',metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_G0 = filter(lambda p: p.requires_grad, model_G0.parameters())
    trainable_params_G1 = filter(lambda p: p.requires_grad, model_G1.parameters())
    trainable_params_G2 = filter(lambda p: p.requires_grad, model_G2.parameters())
    trainable_params_G3 = filter(lambda p: p.requires_grad, model_G3.parameters())
    trainable_params_G4 = filter(lambda p: p.requires_grad, model_G4.parameters())
    trainable_params_G5 = filter(lambda p: p.requires_grad, model_G5.parameters())
    trainable_params_G6 = filter(lambda p: p.requires_grad, model_G6.parameters())
    
    
    ### Optimizers
    optimizer_G0 = config.init_obj('optimizer_G0', torch.optim, trainable_params_G0)
    optimizer_G1 = config.init_obj('optimizer_G1', torch.optim, trainable_params_G1)
    optimizer_G2 = config.init_obj('optimizer_G2', torch.optim, trainable_params_G2)
    optimizer_G3 = config.init_obj('optimizer_G3', torch.optim, trainable_params_G3)
    optimizer_G4 = config.init_obj('optimizer_G4', torch.optim, trainable_params_G4)
    optimizer_G5 = config.init_obj('optimizer_G5', torch.optim, trainable_params_G5)
    optimizer_G6 = config.init_obj('optimizer_G6', torch.optim, trainable_params_G6)
    


    if config['trainer']['criterion']=='MSE':
        criterion = torch.nn.MSELoss()
    if config['trainer']['criterion']=='GEV':
        criterion = None
    # criterion = None
    trainer = Trainer_Proformer(model_G0,model_G1, model_G2, model_G3, model_G4, model_G5, model_G6,
                        data_loader, 
                        valid_data_loader, 
                        criterion, 
                        metrics, 
                        config,
                        optimizer_G0, optimizer_G1, optimizer_G2, optimizer_G3, optimizer_G4,optimizer_G5, optimizer_G6,
                        device=device)
    trainer.train_Proformer()
    

if __name__ == '__main__':
    # ArgumentParser
    args = argparse.ArgumentParser(description='Proformer')
    
    
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r_G0', '--resume_G0', default=None, type=str,
                      help='path to latest checkpoint (Model0) (default: None)')
    args.add_argument('-r_G1', '--resume_G1', default=None, type=str,
                      help='path to latest checkpoint (Model1) (default: None)')
    args.add_argument('-r_G2', '--resume_G2', default=None, type=str,
                      help='path to latest checkpoint (Model2) (default: None)')
    args.add_argument('-r_G3', '--resume_G3', default=None, type=str,
                      help='path to latest checkpoint (Model3) (default: None)')
    args.add_argument('-r_G4', '--resume_G4', default=None, type=str,
                      help='path to latest checkpoint (Model4) (default: None)')
    args.add_argument('-r_G5', '--resume_G5', default=None, type=str,
                      help='path to latest checkpoint (Model5) (default: None)')
    args.add_argument('-r_G6', '--resume_G6', default=None, type=str,
                      help='path to latest checkpoint (Model6) (default: None)')
    # args.add_argument('-r_D', '--resume_D', default=None, type=str,
    #                   help='path to latest checkpoint (DISCRIMINATOR) (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--MODEL', '--MODEL_TYPE'], type=str, target='arch_G0;type'),
        CustomArgs(['--OPT_G0', '--OPT_G_lr0'], type=float, target='optimizer_G0;args;lr'),
        CustomArgs(['--OPT_G1', '--OPT_G_lr1'], type=float, target='optimizer_G1;args;lr'),
        CustomArgs(['--OPT_G2', '--OPT_G_lr2'], type=float, target='optimizer_G2;args;lr'),
        CustomArgs(['--OPT_G3', '--OPT_G_lr3'], type=float, target='optimizer_G3;args;lr'),
        CustomArgs(['--OPT_G4', '--OPT_G_lr4'], type=float, target='optimizer_G4;args;lr'),
        CustomArgs(['--OPT_G5', '--OPT_G_lr5'], type=float, target='optimizer_G5;args;lr'),
        CustomArgs(['--OPT_G6', '--OPT_G_lr6'], type=float, target='optimizer_G6;args;lr'),
        CustomArgs(['--D_MOD0', '--D_MODEL0'], type=int, target='arch_G0;args;d_model'),
        CustomArgs(['--D_MOD1', '--D_MODEL1'], type=int, target='arch_G1;args;d_model'),
        CustomArgs(['--D_MOD2', '--D_MODEL2'], type=int, target='arch_G2;args;d_model'),
        CustomArgs(['--D_MOD3', '--D_MODEL3'], type=int, target='arch_G3;args;d_model'),
        CustomArgs(['--D_MOD4', '--D_MODEL4'], type=int, target='arch_G4;args;d_model'),
        CustomArgs(['--D_MOD5', '--D_MODEL5'], type=int, target='arch_G5;args;d_model'),
        CustomArgs(['--D_MOD6', '--D_MODEL6'], type=int, target='arch_G6;args;d_model'),
        CustomArgs(['--E_LAYERS', '--E_LAYERS'], type=int, target='arch_G0;args;e_layers'),
        CustomArgs(['--DFF', '--d_ff'], type=int, target='arch_G0;args;d_ff'),
        CustomArgs(['--N_HEADS', '--N_HEADS'], type=int, target='arch_G0;args;n_heads'),
        CustomArgs(['--D_LAYERS', '--D_LAYERS'], type=int, target='arch_G0;args;d_layers'),
        CustomArgs(['--SEQ_LEN', '--SEQ_LENGTH'], type=int, target='data_loader;args;seq_len'),
        CustomArgs(['--LABEL_LEN', '--LABEL_LENGTH'], type=int, target='data_loader;args;label_len'),
        CustomArgs(['--PRED_LEN', '--PREDICTION_LENGTH'], type=int, target='data_loader;args;pred_len'),
        CustomArgs(['--SEC', '--SECONDS'], type=int, target='data_loader;args;sec')
    ]
    config = ConfigParser_AutoProformer.from_args(args, options)
    print('begin main')
    main(config)
