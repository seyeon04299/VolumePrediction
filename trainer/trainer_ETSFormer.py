from cProfile import label
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer, BaseTrainer_Autoformer
from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN, calculate_fretchet, prepare_device
import torch.nn.functional as F
from utils import bayeLq_loss,bayeGen_loss, bayeLq_loss1, bayeLq_loss_n_ch, Sinogram_loss, bayeLq_Sino_loss, bayeLq_Sino_loss1,sigma_calc
from utils import loss_quantile, plot_classes_preds, plot_classes_preds_with_truth




#########################################################################################################
### Trainer_ETSformer
#########################################################################################################




class Trainer_ETSFormer(BaseTrainer_Autoformer):
    """
    Trainer class
    """
    def __init__(self, model_G, data_loader, valid_data_loader, criterion, metric_ftns, config, optimizer_G, device,
                 lr_scheduler_G=None, len_epoch=None):
        super().__init__(model_G, criterion, metric_ftns, optimizer_G, config)
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
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        self.label_len = self.config['data_loader']['args']['label_len']
        self.pred_len = self.config['data_loader']['args']['pred_len']
        self.use_amp = self.config['trainer']['use_amp']
        self.output_latents = self.config['arch_G']['args']['output_latents']

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        ## Plotting for whole day
        self.sec = self.config['data_loader']['args']['sec']
        self.multiplier = int(5*60/self.sec)
        end_time = self.pred_len+self.config['data_loader']['args']['seq_len']
        if self.sec*end_time==23400:
            self.plot_whole_day_with_5min_int = True
        else:
            self.plot_whole_day_with_5min_int = False
            if self.sec==1 or self.sec==5:
                ## Since whole day plotting is not possible, plot with 10sec interval
                self.multiplier = int(10/self.sec)
            else:
                self.multiplier = int(60/self.sec)



        self.train_metrics = MetricTracker('MSEloss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('MSEloss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

 
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        
        train_len = len(self.data_loader) 
        # valid_len = len(self.valid_data_loader)

        self.train_metrics.reset()
        self.train_loss = []
        for batch_idx, (batch_x, batch_y, _,_) in enumerate(self.data_loader):

            batch_size = batch_x.shape[0]
            

            batch_x = batch_x.to(torch.float32).to(self.device)  
            batch_y = batch_y.to(torch.float32).to(self.device)  
            
            self.optimizer_G.zero_grad()
            # print(batch_y.shape)        # torch.Size([32, 144, 8])
            # print(batch_x.shape)        # torch.Size([32, 96, 8])

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if self.output_latents:
                        outputs = self.model_G(batch_x,self.pred_len)[0]
                    else:
                        outputs = self.model_G(batch_x,self.pred_len)

                    # print('outputs.shape', outputs.shape)       
                    f_dim = 0
                    outputs = outputs[:, -self.pred_len:, f_dim]
                    # print('outputs1.shape', outputs.shape)
                    batch_y = batch_y[:, -self.pred_len:, f_dim].to(self.device)
                    if len(outputs.shape)==3:
                        outputs = torch.squeeze(outputs)
                    MSEloss = self.criterion(outputs, batch_y)
                    self.train_loss.append(MSEloss.item())
                    
            else:
                if self.output_latents:
                    outputs = self.model_G(batch_x,self.pred_len)[0]
                else:
                    outputs = self.model_G(batch_x,self.pred_len)
                # print('outputs.shape', outputs.shape)       #[32, 96, 8])
                f_dim = 0
                outputs = outputs[:, -self.pred_len:, f_dim]
                # print('outputs1.shape', outputs.shape)      #([32, 96, 8] 0
                batch_y = batch_y[:, -self.pred_len:, f_dim].to(self.device)
                if len(outputs.shape)==3:
                    outputs = torch.squeeze(outputs)
                MSEloss = self.criterion(outputs, batch_y)
                self.train_loss.append(MSEloss.item())

            
            # print('outputs.shape', outputs.shape) #torch.Size([16, 300, 21])
            # print('batch_y.shape', batch_y.shape) #torch.Size([16, 300, 21])

            
            if self.use_amp:
                self.scaler.scale(MSEloss).backward()
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                MSEloss.backward()
                self.optimizer_G.step()



            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} MSE_Loss: {:.4f}'.format(
                    epoch,
                    self.epochs,
                    self._progress(batch_idx),
                    MSEloss.item()))
                # self.writer.add_image('Input_Images', make_grid(data.cpu(), nrow=8, normalize=True))


            if batch_idx == self.len_epoch:
                break

        
        self.writer.set_step(self.tensor_step)
        loss = np.average(self.train_loss)
        self.train_metrics.update('MSEloss', loss.item())

        

        log = self.train_metrics.result()
        self.writer.add_figure('Generated_Series_whole', plot_classes_preds_with_truth(torch.squeeze(outputs[:,-self.pred_len:]).cpu().detach().numpy(),torch.squeeze(batch_y[:,-self.pred_len:]).cpu().detach().numpy() ))
        
        outputs = outputs[:,-self.pred_len:]
        batch_y = batch_y[:,-self.pred_len:]
        outputs_5min = torch.sum(outputs.reshape(-1,int(self.multiplier)),axis=1).reshape(outputs.shape[0],-1)
        batch_y_5min = torch.sum(batch_y.reshape(-1,int(self.multiplier)),axis=1).reshape(batch_y.shape[0],-1)
        
        if self.plot_whole_day_with_5min_int:
            self.writer.add_figure('Generated_Series_in5min', plot_classes_preds_with_truth(torch.squeeze(outputs_5min).cpu().detach().numpy(),torch.squeeze(batch_y_5min).cpu().detach().numpy() ))
        else:
            self.writer.add_figure('Generated_Series_in10sec', plot_classes_preds_with_truth(torch.squeeze(outputs_5min).cpu().detach().numpy(),torch.squeeze(batch_y_5min).cpu().detach().numpy() ))
        
        
        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        self.tensor_step+=1    

        # print('gen_imgs shape: ', gen_imgs.shape)
        # print('gen_imgs.data[:9] shape: ', gen_imgs.data[:9].shape)
        
        # if epoch==0 or epoch%1==0:
        
        

        
        # self.writer.add_figure('Real_Series_future', plot_classes_preds(batch_y[:,:].data.squeeze().cpu()))

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_G.eval()
        self.valid_metrics.reset()

        valid_loss = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, _, _) in enumerate(self.valid_data_loader):
                
                batch_x = batch_x.to(torch.float32).to(self.device)  
                batch_y = batch_y.to(torch.float32).to(self.device)  
                

                if self.output_latents:
                    outputs = self.model_G(batch_x,self.pred_len)[0]
                else:
                    outputs = self.model_G(batch_x,self.pred_len)

                # print('outputs.shape', outputs.shape)       
                f_dim = 0
                outputs = outputs[:, -self.pred_len:, f_dim]
                # print('outputs1.shape', outputs.shape)
                batch_y = batch_y[:, -self.pred_len:, f_dim].to(self.device)
                if len(outputs.shape)==3:
                    outputs = torch.squeeze(outputs)


                loss_val = self.criterion(outputs, batch_y)
                valid_loss.append(loss_val.detach().cpu().numpy())
            loss = np.average(valid_loss)

                
        self.writer.set_step(self.tensor_step, 'valid')
        self.valid_metrics.update('MSEloss', loss.item())
        self.writer.add_figure('Generated_Series_valid', plot_classes_preds_with_truth(torch.squeeze(outputs[:,-self.pred_len:]).cpu().detach().numpy(),torch.squeeze(batch_y[:,-self.pred_len:]).cpu().detach().numpy() ))

        outputs_5min = torch.sum(outputs.reshape(-1,int(self.multiplier)),axis=1).reshape(outputs.shape[0],int(outputs.shape[1]/self.multiplier))
        batch_y_5min = torch.sum(batch_y.reshape(-1,int(self.multiplier)),axis=1).reshape(batch_y.shape[0],int(batch_y.shape[1]/self.multiplier))
        
        # print(outputs.shape)
        if self.plot_whole_day_with_5min_int:
            self.writer.add_figure('Generated_Series_in5min_valid', plot_classes_preds_with_truth(torch.squeeze(outputs_5min).cpu().detach().numpy(),torch.squeeze(batch_y_5min).cpu().detach().numpy() ))
        else:
            self.writer.add_figure('Generated_Series_in10sec_valid', plot_classes_preds_with_truth(torch.squeeze(outputs_5min).cpu().detach().numpy(),torch.squeeze(batch_y_5min).cpu().detach().numpy() ))
        

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model_G.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


