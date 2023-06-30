import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN, calculate_fretchet, prepare_device
from data_loader.data_loaders import KNUskinDataLoader_ProGAN
from model import InceptionV3
import torch.nn.functional as F
from utils import bayeLq_loss,bayeGen_loss, bayeLq_loss1, bayeLq_loss_n_ch, Sinogram_loss, bayeLq_Sino_loss, bayeLq_Sino_loss1,sigma_calc





#########################################################################################################
### Trainer_UNCGAN
#########################################################################################################




class Trainer_UNCGAN(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler_G=None,lr_scheduler_D=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model=self.model.to(device)
        _, device_ids = prepare_device(config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        self.train_metrics = MetricTracker('loss_D', 'FID','avg_tot_loss','avg_sigma', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

 
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()
        avg_rec_loss=0
        avg_tot_loss=0
        avg_tot_loss_tmp=0
        loss_D_tmp=0
        fretchet_dist_tmp=0
        mean_sigma_tmp=0
        self.train_metrics.reset()
        for batch_idx, (data1,data2) in enumerate(self.data_loader):

            data1 = data1.to(self.device)
            data2 = data2.to(self.device)

            rec_B, rec_alpha_B, rec_beta_B = self.model_G(data1)

            # print(rec_B.shape)              # (8,3,224,224)
            # print(rec_alpha_B.shape)        # (1,3,224,224)
            # print(rec_beta_B.shape)         # (1,3,224,224)

            #first gen
            self.model_D.eval()
            total_loss = self.lam1*F.l1_loss(rec_B, data2) + self.lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, data2)
            t0 = self.model_D(rec_B)
            t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(self.device).type(torch.cuda.FloatTensor))
            total_loss += e5
            self.optimizer_G.zero_grad()
            total_loss.backward()
            self.optimizer_G.step()


            #then discriminator
            self.model_D.train()
            t0 = self.model_D(data2)
            pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_real = 1*F.mse_loss(
                pred_real_A, torch.ones(pred_real_A.size()).to(self.device).type(torch.cuda.FloatTensor)
            )
            t0 = self.model_D(rec_B.detach())
            pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_pred = 1*F.mse_loss(
                pred_fake_A, torch.zeros(pred_fake_A.size()).to(self.device).type(torch.cuda.FloatTensor)
            )
            loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5


            loss_D = loss_D_A
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

            avg_tot_loss_tmp += total_loss.item()
            loss_D_tmp += loss_D

            fretchet_dist=calculate_fretchet(data2,rec_B,self.model)
            fretchet_dist_tmp += fretchet_dist

            sigma_uncertainty = sigma_calc(rec_alpha_B,rec_beta_B)
            mean_sigma = torch.mean(sigma_uncertainty)
            mean_sigma_tmp += mean_sigma
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))





            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} total_loss: {:.4f}, Loss_D: {:.4f}, FID: {:.4f}, avg_sig: {: 4f}'.format(
                    epoch,
                    self.num_epochs,
                    self._progress(batch_idx),
                    total_loss.item(),
                    loss_D.item(),
                    fretchet_dist,
                    mean_sigma))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        avg_tot_loss = avg_tot_loss_tmp/len(self.data_loader)
        loss_D = loss_D_tmp/len(self.data_loader)
        fretchet_dist = fretchet_dist_tmp/len(self.data_loader)
        avg_sigma = mean_sigma_tmp/len(self.data_loader)
        
        
        self.writer.set_step(self.tensor_step)
        self.train_metrics.update('loss_D', loss_D.item())
        self.train_metrics.update('FID', fretchet_dist)
        self.train_metrics.update('avg_tot_loss', avg_tot_loss)
        self.train_metrics.update('avg_sigma', avg_sigma)

        self.writer.add_scalar('avg_sigma', avg_sigma)
        self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

        # print('gen_imgs shape: ', gen_imgs.shape)
        # print('gen_imgs.data[:9] shape: ', gen_imgs.data[:9].shape)
        
        # if epoch==0 or epoch%1==0:
        self.writer.add_image('Generated_Images', make_grid(rec_B.data.cpu(), normalize=True))
        self.writer.add_image('rec_alpha_B', make_grid(rec_alpha_B.data.cpu(), ormalize=True))
        self.writer.add_image('rec_beta_B', make_grid(rec_beta_B.data.cpu(), normalize=True))
        
        # if epoch==1:
        # Imput Image
        self.writer.add_image('Noisy_Inputs', make_grid(data1.cpu(), normalize=True))
        self.writer.add_image('Melanoma_Inputs', make_grid(data2.cpu(), normalize=True))
        self.writer.add_image('Uncertainty_map', make_grid(sigma_uncertainty.cpu(), normalize=True))
        return log

    


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)




class Trainer_UNCGAN_M(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_G0, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler_G=None,lr_scheduler_D=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        self.model_G0 = model_G0

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model=self.model.to(device)
        _, device_ids = prepare_device(config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        self.train_metrics = MetricTracker('loss_D', 'FID','avg_tot_loss', 'avg_sigma', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

 
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()
        avg_rec_loss=0
        avg_tot_loss=0
        avg_tot_loss_tmp=0
        loss_D_tmp=0
        fretchet_dist_tmp=0
        mean_sigma_tmp=0
        self.train_metrics.reset()
        for batch_idx, (data1,data2) in enumerate(self.data_loader):

            data1 = data1.to(self.device)
            data2 = data2.to(self.device)

            
            rec_B, rec_alpha_B, rec_beta_B = self.model_G0(data1)
            
            xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, data1], dim=1)
            rec_B, rec_alpha_B, rec_beta_B = self.model_G(xch)

            # print(rec_B.shape)              # (8,3,224,224)
            # print(rec_alpha_B.shape)        # (1,3,224,224)
            # print(rec_beta_B.shape)         # (1,3,224,224)

            #first gen
            self.model_D.eval()
            total_loss = self.lam1*F.l1_loss(rec_B, data2) + self.lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, data2)
            t0 = self.model_D(rec_B)
            t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(self.device).type(torch.cuda.FloatTensor))
            total_loss += e5
            self.optimizer_G.zero_grad()
            total_loss.backward()
            self.optimizer_G.step()


            #then discriminator
            self.model_D.train()
            t0 = self.model_D(data2)
            pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_real = 1*F.mse_loss(
                pred_real_A, torch.ones(pred_real_A.size()).to(self.device).type(torch.cuda.FloatTensor)
            )
            t0 = self.model_D(rec_B.detach())
            pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_pred = 1*F.mse_loss(
                pred_fake_A, torch.zeros(pred_fake_A.size()).to(self.device).type(torch.cuda.FloatTensor)
            )
            loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5


            loss_D = loss_D_A
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

            avg_tot_loss_tmp += total_loss.item()
            loss_D_tmp += loss_D

            fretchet_dist=calculate_fretchet(data2,rec_B,self.model)
            fretchet_dist_tmp += fretchet_dist

            sigma_uncertainty = sigma_calc(rec_alpha_B,rec_beta_B)
            mean_sigma = torch.mean(sigma_uncertainty)
            mean_sigma_tmp += mean_sigma
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))





            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} total_loss: {:.4f}, Loss_D: {:.4f}, FID: {:.4f}, avg_sig: {:4f}'.format(
                    epoch,
                    self.num_epochs,
                    self._progress(batch_idx),
                    total_loss.item(),
                    loss_D.item(),
                    fretchet_dist,
                    mean_sigma))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        avg_tot_loss = avg_tot_loss_tmp/len(self.data_loader)
        loss_D = loss_D_tmp/len(self.data_loader)
        fretchet_dist = fretchet_dist_tmp/len(self.data_loader)
        avg_sigma = mean_sigma_tmp/len(self.data_loader)
        
        
        self.writer.set_step(self.tensor_step)
        self.train_metrics.update('loss_D', loss_D.item())
        self.train_metrics.update('FID', fretchet_dist)
        self.train_metrics.update('avg_tot_loss', avg_tot_loss)
        self.train_metrics.update('avg_sigma', avg_sigma)

        self.writer.add_scalar('avg_sigma', avg_sigma)
        self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

        # print('gen_imgs shape: ', gen_imgs.shape)
        # print('gen_imgs.data[:9] shape: ', gen_imgs.data[:9].shape)
        
        # if epoch==0 or epoch%1==0:
        self.writer.add_image('Generated_Images', make_grid(rec_B.data.cpu(), normalize=True))
        self.writer.add_image('rec_alpha_B', make_grid(rec_alpha_B.data.cpu(), normalize=True))
        self.writer.add_image('rec_beta_B', make_grid(rec_beta_B.data.cpu(), normalize=True))
        
        # if epoch==1:
                # Imput Image
        self.writer.add_image('Noisy_Inputs', make_grid(data1.cpu(), normalize=True))
        self.writer.add_image('Melanoma_Inputs', make_grid(data2.cpu(), normalize=True))
        self.writer.add_image('Uncertainty_map', make_grid(sigma_uncertainty.cpu(), normalize=True))
        return log

    


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)





#########################################################################################################
### Trainer_DCGAN
#########################################################################################################


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.data_loader.batch_size))
        # print(self.data_loader.batch_size)
        # print(self.log_step)


        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model=self.model.to(device)
        _, device_ids = prepare_device(config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        self.train_metrics = MetricTracker('loss_G','loss_D', 'FID', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        self.train_metrics.reset()
        for batch_idx, (data,_) in enumerate(self.data_loader):

            data = data.to(self.device)

            valid = torch.full((data.shape[0],1,1,1),1).view(-1,1,1,1).to(self.device).float()
            fake = torch.full((data.shape[0],1,1,1),0).view(-1,1,1,1).to(self.device).float()

            z = torch.normal(0, 1, size=(data.shape[0], self.nz,1,1)).to(self.device)

            # ----------------------------------
            #  Train Generator
            # ----------------------------------
            self.optimizer_G.zero_grad()

            # Feed Forward
            gen_imgs = self.model_G(z)
            # Loss Measures generators ability to fool the discriminator
            loss_G = self.criterion(self.model_D(gen_imgs), valid)
            # Backpropagation
            loss_G.backward()
            self.optimizer_G.step()

            # --------------------------------------
            #  Train Discriminator
            # --------------------------------------
            self.optimizer_D.zero_grad()

            # Feed forward
            y_real = self.model_D(data)
            y_fake = self.model_D(gen_imgs.detach())
            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.criterion(y_real,valid)
            fake_loss = self.criterion(y_fake,fake)
            loss_D = (real_loss+fake_loss)/2
            # Backpropagation
            loss_D.backward()
            self.optimizer_D.step()

            fretchet_dist=calculate_fretchet(data,gen_imgs,self.model)
            


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            self.train_metrics.update('FID', fretchet_dist)

            # TRACK loss_D (REAL), loss_D (Fake)
            self.writer.add_scalar('loss_D_REAL', real_loss)
            self.writer.add_scalar('loss_D_FAKE', fake_loss)
            self.writer.add_scalar('FID', fretchet_dist)
            

            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss_G: {:.4f}, Loss_D: {:.4f}, FID: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item(),
                    fretchet_dist))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None: #None임
            self.lr_scheduler.step()
        # print('gen_imgs shape: ', gen_imgs.shape)
        # print('gen_imgs.data[:9] shape: ', gen_imgs.data[:9].shape)
        
        if epoch==0 or epoch%5==0:
            self.writer.add_image('Generated_Images', make_grid(gen_imgs.data[:6].cpu(), nrow=3, normalize=True))

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)



#########################################################################################################
### Trainer_WGAN, WGAN_GP
#########################################################################################################




class Trainer_WGAN(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        self.critic_iter = self.config['trainer']['critic_iter']

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model=self.model.to(device)
        _, device_ids = prepare_device(config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        if 'WGANGP' == self.model_name:                                  ## WGAN gradient penalty
            self.lambda_GP = self.config['trainer']['lambda_GP']
        if 'WGAN' == self.model_name:                                    ## WGAN clipping
            self.weight_cliping_limit = self.config['trainer']['weight_cliping_limit']


        self.train_metrics = MetricTracker('loss_G','loss_D', 'FID', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        self.train_metrics.reset()

        for batch_idx, (data,_) in enumerate(self.data_loader):

            real = data.to(self.device)

            if 'WGANGP' in self.type_name: 

                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):

                    z = torch.normal(0, 1, size=(real.shape[0], self.nz,1,1)).to(self.device)

                    # Train Discriminator - WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    fake = self.model_G(z)
                    d_loss_real = self.model_D(real).reshape(-1)
                    d_loss_fake = self.model_D(fake).reshape(-1)
                    gp = gradient_penalty(self.model_D,real,fake,device=self.device)
                    loss_D = (
                        -(torch.mean(d_loss_real)-torch.mean(d_loss_fake)) + self.lambda_GP*gp
                    )

                    self.model_D.zero_grad()
                    loss_D.backward(retain_graph=True)      # Reutilize the computations for 'fake' when we do the updates for the generator
                    self.optimizer_D.step()

                ### Train Generator: min -E[D(gen_fake)]
                output = self.model_D(fake).reshape(-1)
                loss_G = -torch.mean(output)
                self.model_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()

            else: # if model==WGAN
                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):

                    z = torch.normal(0, 1, size=(real.shape[0], self.nz,1,1)).to(self.device)

                    # Train Discriminator - WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    fake = self.model_G(z)
                    d_loss_real = self.model_D(real).reshape(-1)
                    d_loss_fake = self.model_D(fake).reshape(-1)
                    loss_D = -(torch.mean(d_loss_real)-torch.mean(d_loss_fake))

                    self.model_D.zero_grad()
                    loss_D.backward(retain_graph=True)      # Reutilize the computations for 'fake' when we do the updates for the generator
                    self.optimizer_D.step()

                    # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                    for p in self.model_D.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                ### Train Generator: min -E[D(gen_fake)]
                output = self.model_D(fake).reshape(-1)
                loss_G = -torch.mean(output)
                self.model_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
                
                

            fretchet_dist=calculate_fretchet(real,fake,self.model)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            self.train_metrics.update('FID', fretchet_dist)
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss_G: {:.4f}, Loss_D: {:.4f}, FID: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item(),
                    fretchet_dist))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            # if batch_idx == self.len_epoch:
            #     break
        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None: #None임
            self.lr_scheduler.step()

        if epoch==0 or epoch%5==0:
            self.writer.add_image('Generated_Images', make_grid(fake.data[:6].cpu(), nrow=3, normalize=True))
        
        if epoch==1:
                # Imput Image
                self.writer.add_image('Input_Images', make_grid(data[:16].cpu(), nrow=4, normalize=True))

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)



#########################################################################################################
### Trainer_ProGAN
#########################################################################################################



class Trainer_ProGAN(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader_dummy = data_loader
        
        # if len_epoch is None:
        #     # epoch-based training
        #     self.len_epoch = len(self.data_loader_dummy)
        # else:
        #     # iteration-based training
        #     self.data_loader = inf_loop(data_loader)
        #     self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        # Set Gradient Penalty Parameter
        self.lambda_GP = self.config['trainer']['lambda_GP']


        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model=self.model.to(device)
        _, device_ids = prepare_device(config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        dlp = config['data_loader']['args']        # for calling dataloader parameters
        self.dlp_data_dir = dlp['data_dir']
        self.dlp_subject_map = dlp['subject_map']
        self.dlp_cell_type = dlp['cell_type']
        self.dlp_shuffle = dlp['shuffle']
        self.dlp_validation_split = dlp['validation_split']
        self.dlp_num_workers = dlp['num_workers']

        self.fixed_noise = torch.normal(0, 1, size=(16, self.nz,1,1)).to(self.device)


        self.train_metrics = MetricTracker('loss_G','loss_D', 'FID', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        self.train_metrics.reset()

        if self.use_autocast:
            scaler_D = torch.cuda.amp.GradScaler()
            scaler_G = torch.cuda.amp.GradScaler()

        ### GET DATALOADER according to img size, batch size
        self.data_loader = KNUskinDataLoader_ProGAN(
            data_dir=self.dlp_data_dir,
            batch_size=self.batch_size,
            input_size=self.img_size,
            subject_map=self.dlp_subject_map,
            cell_type=self.dlp_cell_type,
            shuffle=self.dlp_shuffle,
            validation_split=self.dlp_validation_split,
            num_workers=self.dlp_num_workers)
        
        
        self.len_epoch = len(self.data_loader)

        for batch_idx, (data,_) in enumerate(self.data_loader):
            # print('alpha : ', self.alpha)
            real = data.to(self.device)

            z = torch.normal(0, 1, size=(real.shape[0], self.nz,1,1)).to(self.device)

            if self.use_autocast:
                with torch.cuda.amp.autocast():
                # Train Discriminator : MAX (E[critic(real)]-E[critic(fake)])
                    fake = self.model_G(z, self.alpha, self.step)
                    d_loss_real = self.model_D(real,self.alpha,self.step).reshape(-1)
                    d_loss_fake = self.model_D(fake.detach(),self.alpha, self.step)
                    gp = gradient_penalty_ProGAN(self.model_D, real, fake, self.alpha, self.step, device=self.device)
                    loss_D = (
                        -(torch.mean(d_loss_real)-torch.mean(d_loss_fake))
                        + self.lambda_GP*gp
                        + (0.001 * torch.mean(d_loss_real**2))
                    )

                self.optimizer_D.zero_grad()
                scaler_D.scale(loss_D).backward(retain_graph=True)
                scaler_D.step(self.optimizer_D)
                scaler_D.update()


                # Train Generator: MAX E[D(gen_fake)]  <=> MIN -E[D(gen_fake)]
                with torch.cuda.amp.autocast():
                    output = self.model_D(fake,self.alpha,self.step)
                    loss_G = -torch.mean(output)

                self.optimizer_G.zero_grad()
                scaler_G.scale(loss_G).backward()
                scaler_G.step(self.optimizer_G)
                scaler_G.update()

                
            
            else: ## Not use autocast
                
                # Train Discriminator : MAX (E[critic(real)]-E[critic(fake)])
                fake = self.model_G(z, self.alpha, self.step)
                d_loss_real = self.model_D(real,self.alpha,self.step).reshape(-1)
                d_loss_fake = self.model_D(fake.detach(),self.alpha, self.step)
                gp = gradient_penalty_ProGAN(self.model_D, real, fake, self.alpha, self.step, device=self.device)
                loss_D = (
                    -(torch.mean(d_loss_real)-torch.mean(d_loss_fake))
                    + self.lambda_GP*gp
                    + (0.001 * torch.mean(d_loss_real**2))
                )
                
                self.model_D.zero_grad()
                loss_D.backward(retain_graph=True)
                self.optimizer_D.step()
                
                # Train Generator: MAX E[D(gen_fake)]  <=> MIN -E[D(gen_fake)]
                output = self.model_D(fake,self.alpha,self.step)
                loss_G = -torch.mean(output)
                
                self.model_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
                
                
            # print('realshape : ', real.shape[0])
            # print('len(self.data_loader.dataset) : ', len(self.data_loader.dataset))
            self.alpha += real.detach().shape[0]/ (len(self.data_loader.dataset)*self.progressive_epochs[self.step]*0.3)
            self.alpha = min(self.alpha,1)
            # print('self.alpha += real.shape[0]/ len(self.data_loader.dataset) = ',self.alpha)
            

            self.writer.set_step(self.tensor_step)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            
            
            fretchet_dist=calculate_fretchet(real,fake.to(dtype=torch.float32),self.model)
            self.train_metrics.update('FID', fretchet_dist)
            
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch {} (ImgSize: {}) {} Loss_G: {:.4f}, Loss_D: {:.4f}, FID: {:.4f}'.format(
                    epoch,
                    self.img_size,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item(),
                    fretchet_dist))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            # if batch_idx == self.len_epoch:
            #     break
            
            self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None: #None임
            self.lr_scheduler.step()

        if epoch==1 or epoch%5==0:
            with torch.no_grad():
                random_fake = self.model_G(z, self.alpha, self.step)
                fixed_fake = self.model_G(self.fixed_noise, self.alpha, self.step)
                # Generated Random Image
                self.writer.add_image('Generated_Images', make_grid(random_fake.data[:16].cpu(), nrow=4, normalize=True))
                # Generated Fixed Image
                self.writer.add_image('Generated_Images (Fixed)', make_grid(fixed_fake.data[:16].cpu(), nrow=4, normalize=True))
                if epoch==1:
                    # Imput Image
                    self.writer.add_image('Input_Images', make_grid(data[:16].cpu(), nrow=4, normalize=True))

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

