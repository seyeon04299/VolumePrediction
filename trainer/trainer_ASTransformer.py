from cProfile import label
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer, BaseTrainer_ASTransformer
from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN, calculate_fretchet, prepare_device
import torch.nn.functional as F
from utils import bayeLq_loss,bayeGen_loss, bayeLq_loss1, bayeLq_loss_n_ch, Sinogram_loss, bayeLq_Sino_loss, bayeLq_Sino_loss1,sigma_calc
from utils import loss_quantile, plot_classes_preds




#########################################################################################################
### Trainer_ASTransfomer
#########################################################################################################




class Trainer_ASTransfomer(BaseTrainer_ASTransformer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, data_loader, valid_data_loader, criterion, metric_ftns, config, optimizer_G, optimizer_D, device,
                 lr_scheduler_G=None,lr_scheduler_D=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.criterion = criterion
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

        self.predict_start = self.config['data_loader']['args']['p']
        self.gan = self.config['trainer']['gan']
        self.Wloss = self.config['trainer']['Wloss']
        self.lambda_GP = self.config['trainer']['lambda_GP']
        self.train_window = self.config['data_loader']['args']['p'] + self.config['data_loader']['args']['q']
        self.lmbda = self.config['trainer']['lmbda']



        self.train_metrics = MetricTracker('loss_G', 'loss_D', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

 
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        best_valid_q50 = float('inf')
        best_valid_q90 = float('inf')
        best_test_q50 = float('inf')
        best_test_q90 = float('inf')
        best_MAPE = float('inf')
        train_len = len(self.data_loader) 
        # valid_len = len(self.valid_data_loader)
    
        loss_epoch = 0
        d_loss_epoch = 0
        # e_loss_epoch = 0

        mean_sigma_tmp=0
        self.train_metrics.reset()

        for batch_idx, (train_batch,idx,labels_batch) in enumerate(self.data_loader):

            batch_size = train_batch.shape[0]

            train_batch = train_batch.to(torch.float32).to(self.device)  
            labels_batch = labels_batch.to(torch.float32).to(self.device)  
            idx = idx.unsqueeze(-1).to(self.device) 


            # Adversarial ground truths
            valid = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            labels = labels_batch[:,self.predict_start:]


            q50, q90 = self.model_G(train_batch,idx)



            # print('q50.shape : ', q50.shape)
            # print('q90.shape : ', q90.shape)
            # print('data.shape: ', train_batch.shape)
            # print('fake_input: ', torch.cat((labels_batch[:,:self.predict_start], q50), 1).shape)

            d_loss = 0

            if self.gan=='False':
                self.optimizer_G.zero_grad()
                loss_50 = loss_quantile(q50, labels, torch.tensor(0.5)) 
                loss_50.backward()
                self.optimizer_G.step()
                g_loss_50 = loss_50.item() / self.train_window
                loss_epoch += g_loss
            else:
                
                # fake_input = torch.cat((torch.squeeze(labels_batch[:,:self.predict_start]), q50), 1)
                fake_input = q50
                
                
                # print(fake_input.shape)

                if self.Wloss:
                    #-------------------------------------------------------------------
                    # Train the generator 
                    #-------------------------------------------------------------------
                    self.optimizer_G.zero_grad()
                    output_D = self.model_D(fake_input).reshape(-1)
                    
                    loss = loss_quantile(q50, labels, torch.tensor(0.5)) + self.lmbda * (-torch.mean(output_D))
                    self.optimizer_G.step()
                    g_loss = loss / self.train_window
                    loss_epoch += g_loss

                    #-------------------------------------------------------------------
                    # Train the discriminator
                    # Forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                    #-------------------------------------------------------------------
                    # for d_iter in range(self.critic_iter):
                    self.optimizer_D.zero_grad()
                    real_loss = self.model_D(torch.squeeze(labels)).reshape(-1)
                    # real_loss = self.model_D(torch.squeeze(labels_batch)).reshape(-1)
                    fake_loss = self.model_D(fake_input).reshape(-1)

                    # print(labels_batch.shape)           # torch.Size([32, 600, 1])
                    # print(fake_input.shape)             # torch.Size([32, 600])

                    # gp = gradient_penalty(self.model_D,torch.squeeze(labels_batch),fake_input,device=self.device)
                    gp = gradient_penalty(self.model_D,torch.squeeze(labels),fake_input,device=self.device)
                    loss_d = (
                        -(torch.mean(real_loss)-torch.mean(fake_loss)) + self.lambda_GP*gp
                    )
                    loss_d.backward()
                    self.optimizer_D.step()  
                        
                    d_loss = loss_d
                    d_loss_epoch += d_loss




                else:
                    #-------------------------------------------------------------------
                    # Train the generator 
                    #-------------------------------------------------------------------
                    self.optimizer_G.zero_grad()
                    loss =  loss_quantile(q50, labels, torch.tensor(0.5)) + 0.1 * self.criterion(self.model_D(fake_input), valid)
                    
                    loss.backward()
                    self.optimizer_G.step()
                    g_loss = loss / self.train_window
                    loss_epoch += g_loss

                    #-------------------------------------------------------------------
                    # Train the discriminator
                    #-------------------------------------------------------------------
                    self.optimizer_D.zero_grad()
                    # real_loss = self.criterion(self.model_D(torch.squeeze(labels_batch)), valid)
                    real_loss = self.criterion(self.model_D(torch.squeeze(labels)), valid)
                    fake_loss = self.criterion(self.model_D(fake_input.detach()), fake)
                    loss_d = 0.5*(real_loss + fake_loss)
                    loss_d.backward()
                    self.optimizer_D.step()  
                        
                    d_loss = loss_d
                    d_loss_epoch += d_loss


            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} Loss_G: {:.4f}, Loss_D: {:.4f}'.format(
                    epoch,
                    self.epochs,
                    self._progress(batch_idx),
                    g_loss.item(),
                    d_loss.item()))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))


            if batch_idx == self.len_epoch:
                break

        
        avg_loss = loss_epoch/len(self.data_loader)
        avg_dloss = d_loss_epoch/len(self.data_loader)

        self.writer.set_step(self.tensor_step)
        self.train_metrics.update('loss_D', avg_dloss.item())
        self.train_metrics.update('loss_G', avg_loss.item())

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
        self.writer.add_figure('Generated_Series_future', plot_classes_preds(fake_input.data.squeeze().cpu()))
        self.writer.add_figure('Generated_Series_q90', plot_classes_preds(torch.cat((torch.squeeze(labels_batch[:,:self.predict_start]), q90), 1).data.squeeze().cpu()))
        self.writer.add_figure('Generated_Series_whole', plot_classes_preds(torch.cat((torch.squeeze(labels_batch[:,:self.predict_start]), q50), 1).data.squeeze().cpu()))
        self.writer.add_figure('Real_Series_future', plot_classes_preds(labels.data.squeeze().cpu()))
        self.writer.add_figure('Real_Series_Whole', plot_classes_preds(labels_batch.data.squeeze().cpu()))

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


