import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer, BaseTrainer_TimeGAN
from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN, calculate_fretchet, prepare_device
from data_loader.data_loaders import DataLoader_VolumeAlloc
import torch.nn.functional as F
from utils import plot_classes_preds, bayeLq_loss,bayeGen_loss, bayeLq_loss1, bayeLq_loss_n_ch, Sinogram_loss, bayeLq_Sino_loss, bayeLq_Sino_loss1,sigma_calc





#########################################################################################################
### Trainer_TimeGAN
#########################################################################################################




class Trainer_TimeGAN(BaseTrainer_TimeGAN):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, config, device, 
                data_loader, valid_data_loader=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, config)
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

        self.lr_scheduler_e = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_e)
        self.lr_scheduler_r = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_r)
        self.lr_scheduler_s = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_s)
        self.lr_scheduler_g = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_g)
        self.lr_scheduler_d = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_d)
        
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']
        self.dis_thresh = self.config['trainer']['dis_thresh']
        self.gamma = self.config['trainer']['gamma']
        self.Z_dim = self.config['arch']['args']['Z_dim']
        self.p = self.config['data_loader']['args']['p']
        self.q = self.config['data_loader']['args']['q']

        # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        # self.model = InceptionV3([block_idx])
        # self.model=self.model.to(device)
        # _, device_ids = prepare_device(config['n_gpu'])
        # self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        
        self.train_metrics = MetricTracker('embedding_loss', 'supervisor_loss','joint_embedding_loss','joint_generator_loss','joint_dicriminator_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch_emb(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        for batch_idx, (X_mb,T_mb, max_len) in enumerate(self.data_loader):
            # Reset gradients
            self.model.zero_grad()
            # X_mb_future = X_mb[:,self.p:self.q,:]
            X_mb_whole = X_mb.to(self.device).float()
            X_mb = X_mb[:,:self.p,:].to(self.device).float()
            # print(X_mb.shape) # torch.Size([32, 300, 1])
            # T_mb = T_mb.to(self.device)
            

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            H, X_tilde,H_hat_supervise = self.model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")

            G_loss_S = torch.nn.functional.mse_loss(
                H_hat_supervise[:,:-1,:], 
                H[:,1:,:]
            ) # Teacher forcing next output

            # Reconstruction Loss
            E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X_mb)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S
            # return E_loss, E_loss0, E_loss_T0
            # _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())

            # Backward Pass
            E_loss0.backward()

            # Update model parameters
            self.optimizer_e.step()
            self.optimizer_r.step()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} || Embedding/Loss: {:.4f}'.format(
                    epoch,
                    self.epochs_emb,
                    self._progress(batch_idx),
                    loss))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        self.writer.set_step(epoch)
        self.train_metrics.update('embedding_loss', loss)

        self.writer.add_scalar('embedding_loss', loss)
        self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_e is not None:
            self.lr_scheduler_e.step()
        if self.lr_scheduler_r is not None:
            self.lr_scheduler_r.step()

        return log


    def _train_epoch_sup(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        for batch_idx, (X_mb,T_mb, max_len) in enumerate(self.data_loader):
            # Reset gradients
            self.model.zero_grad()

            # X_mb_future = X_mb[:,self.p:self.q,:]
            X_mb_whole = X_mb.to(self.device).float()
            X_mb = X_mb[:,:self.p,:].to(self.device).float()



            # X_mb = X_mb.to(self.device).float()
            # T_mb = T_mb.to(self.device)

            # Forward Pass
            H, H_hat_supervise = self.model(X=X_mb, T=T_mb, Z=None, obj="supervisor")

            # Supervised loss
            S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output
            
            # Backward Pass
            S_loss.backward()
            loss = np.sqrt(S_loss.item())

            # Update model parameters
            self.optimizer_s.step()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} || Supervisor/Loss: {:.4f}'.format(
                    epoch,
                    self.epochs_sup,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        
        self.writer.set_step(epoch)
        self.train_metrics.update('supervisor_loss', loss)


        self.writer.add_scalar('supervisor_loss', loss)
        self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_s is not None:
            self.lr_scheduler_s.step()

        return log
        
    
    def _train_epoch_joint(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        """The training loop for training the model altogether
        """
        self.model.train()

        for batch_idx, (X_mb,T_mb, max_len) in enumerate(self.data_loader):
            # Reset gradients
            self.model.zero_grad()

            # X_mb_future = X_mb[:,self.p:self.q,:]
            X_mb_whole = X_mb.to(self.device).float()
            X_mb = X_mb[:,:self.p,:].to(self.device).float()
            # X_mb = X_mb.to(self.device).float()
            # T_mb = T_mb.to(self.device)

            ## Generator Training
            for _ in range(2):
                # Random Generator
                Z_mb = torch.normal(0, 1, size=(X_mb.shape[0], X_mb.shape[1],self.Z_dim)).float().to(self.device) # Originally, args.Z_dim=1
                # Z_mb = torch.normal(0, 1, size=(X_mb.shape[0], self.nz,1,1)).to(self.device)
                # Z_mb = torch.rand((self.data_loader.batch_size, args.max_seq_len, args.Z_dim))

                # Forward Pass (Generator)
                self.model.zero_grad()
                H,H_hat_supervise,E_hat,H_hat,X_hat,Y_fake,Y_fake_e = self.model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")

                # Generator Loss
                # 1. Adversarial loss
                G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

                # 2. Supervised loss
                G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

                # 3. Two Momments
                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X_mb.var(dim=0, unbiased=False) + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X_mb.mean(dim=0))))

                G_loss_V = G_loss_V1 + G_loss_V2

                # 4. Summation
                G_loss = G_loss_U + self.gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                self.optimizer_g.step()
                self.optimizer_s.step()

                # Forward Pass (Embedding)
                self.model.zero_grad()
                H, X_tilde,H_hat_supervise = self.model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                
                G_loss_S = torch.nn.functional.mse_loss(
                    H_hat_supervise[:,:-1,:], 
                    H[:,1:,:]
                ) # Teacher forcing next output

                # Reconstruction Loss
                E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X_mb)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * G_loss_S

                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())
                
                # Update model parameters
                self.optimizer_e.step()
                self.optimizer_r.step()

            # Random Generator
            Z_mb = torch.normal(0, 1, size=(X_mb.shape[0], X_mb.shape[1],self.Z_dim)).float().to(self.device) # Originally, args.Z_dim=1
            # Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

            ## Discriminator Training
            self.model.zero_grad()
            # Forward Pass
            H,E_hat,H_hat,Y_real,Y_fake,Y_fake_e = self.model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

            D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
            D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e

            # Check Discriminator loss (train discriminator only when the discriminator does not work well)
            if D_loss > self.dis_thresh:
                # Backward Pass
                D_loss.backward()

                # Update model parameters
                self.optimizer_d.step()
            D_loss = D_loss.item()




            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} ||  Joint/Embedding_Loss: {:.4f}, Joint/Generator_Loss: {:.4f}, Joint/Discriminator_Loss: {:.4f}'.format(
                    epoch,
                    self.epochs_joint,
                    self._progress(batch_idx),
                    E_loss,
                    G_loss,
                    D_loss))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        Z_mb = torch.normal(0, 1, size=(X_mb.shape[0],  X_mb.shape[1],self.Z_dim)).float().to(self.device)
        X_hat = self.model(X=X_mb, T=T_mb, Z=Z_mb, obj="inference")

        self.writer.set_step(epoch)
        self.train_metrics.update('joint_embedding_loss', E_loss)
        self.train_metrics.update('joint_generator_loss', G_loss)
        self.train_metrics.update('joint_dicriminator_loss', D_loss)

        self.writer.add_scalar('joint_embedding_loss', E_loss)
        self.writer.add_scalar('joint_generator_loss', G_loss)
        self.writer.add_scalar('joint_dicriminator_loss', D_loss)

        
        self.writer.add_figure('Generated_Series', plot_classes_preds(X_hat.data.squeeze().cpu()))
        self.writer.add_figure('Real_Series', plot_classes_preds(X_mb_whole[:,1:,:].data.squeeze().cpu()))
        self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_s is not None:
            self.lr_scheduler_s.step()
        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step()
        if self.lr_scheduler_e is not None:
            self.lr_scheduler_e.step()
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()
        if self.lr_scheduler_r is not None:
            self.lr_scheduler_r.step()

        # print('gen_imgs shape: ', gen_imgs.shape)
        # print('gen_imgs.data[:9] shape: ', gen_imgs.data[:9].shape)
        
        # # if epoch==0 or epoch%1==0:
        # self.writer.add_image('Generated_Images', make_grid(rec_B.data.cpu(), normalize=True))
        # self.writer.add_image('rec_alpha_B', make_grid(rec_alpha_B.data.cpu(), ormalize=True))
        # self.writer.add_image('rec_beta_B', make_grid(rec_beta_B.data.cpu(), normalize=True))
        
        # # if epoch==1:
        # # Imput Image
        # self.writer.add_image('Noisy_Inputs', make_grid(data1.cpu(), normalize=True))
        # self.writer.add_image('Melanoma_Inputs', make_grid(data2.cpu(), normalize=True))
        # self.writer.add_image('Uncertainty_map', make_grid(sigma_uncertainty.cpu(), normalize=True))
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


