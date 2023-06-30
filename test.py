import argparse
import torch
from tqdm import tqdm
from datetime import datetime
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        subject_map="data/KNU_DSLR/KNU_DSLR_subjectmap.csv"
    )

    # build model architecture
    model_G = config.init_obj('arch_G', module_arch)
    logger.info(model_G)
    model_D = config.init_obj('arch_D', module_arch)
    logger.info(model_D)

    # get function handles of loss and metrics
    loss_fn_G = getattr(module_loss, config['loss_G'])
    loss_fn_D = getattr(module_loss, config['loss_D'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume_G))
    logger.info('Loading checkpoint: {} ...'.format(config.resume_D))
    checkpoint_G = torch.load(config.resume_G)
    state_dict_G = checkpoint_G['state_dict']
    if config['n_gpu'] > 1:
        model_G = torch.nn.DataParallel(model_G)
    model_G.load_state_dict(state_dict_G)

    checkpoint_D = torch.load(config.resume_D)
    state_dict_D = checkpoint_D['state_dict']
    if config['n_gpu'] > 1:
        model_D = torch.nn.DataParallel(model_D)
    model_D.load_state_dict(state_dict_D)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_G = model_G.to(device)
    model_D = model_D.to(device)
    
    model_G.eval()
    model_D.eval()

    total_loss_G = 0.0
    total_loss_D = 0.0
    
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, loaded_data in enumerate(tqdm(data_loader)):
            data = loaded_data[0]
            target = loaded_data[1]
            if len(loaded_data)==3: # if Sitelabel is used as input
                site = loaded_data[2].float()
                site = site.to(device)
            data, target = data.to(device), target.to(device)
            
            if len(loaded_data)==3:
                output = model(data,site)
            if len(loaded_data)==2:
                output = model(data)
            
            ###
            data = data.to(device)

            valid = torch.full((data.shape[0],1),1).view(-1,1).to(device).float()
            fake = torch.full((data.shape[0],1),0).view(-1,1).to(device).float()

            z = torch.normal(0, 1, size=(data.shape[0], nz,1)).to(device)

            # # ----------------------------------
            # #  Train Generator
            # # ----------------------------------
            # self.optimizer_G.zero_grad()

            # # Feed Forward
            # gen_imgs = self.model_G(z)
            # # Loss Measures generators ability to fool the discriminator
            # loss_G = self.criterion(self.model_D(gen_imgs), valid)
            # # Backpropagation
            # loss_G.backward()
            # self.optimizer_G.step()

            # # --------------------------------------
            # #  Train Discriminator
            # # --------------------------------------
            # self.optimizer_D.zero_grad()

            # # Feed forward
            # y_real = self.model_D(data)
            # y_fake = self.model_D(gen_imgs.detach())
            # # Measure discriminator's ability to classify real from generated samples
            # real_loss = self.criterion(y_real,valid)
            # fake_loss = self.criterion(y_fake,fake)
            # loss_D = (real_loss+fake_loss)/2
            # # Backpropagation
            # loss_D.backward()
            # self.optimizer_D.step()

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)
