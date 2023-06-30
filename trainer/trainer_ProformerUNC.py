from cProfile import label
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer, BaseTrainer_Autoformer, BaseTrainer_Proformer
from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN, calculate_fretchet, prepare_device
import torch.nn.functional as F
from utils import bayeLq_loss,bayeGen_loss, bayeLq_loss1, bayeLq_loss_n_ch, Sinogram_loss, bayeLq_Sino_loss, bayeLq_Sino_loss1,sigma_calc
from utils import loss_quantile, plot_classes_preds, plot_classes_preds_with_truth
from utils import unnormalize_historic, unnormalize_then_normalize




#########################################################################################################
### Trainer_Proformer
#########################################################################################################




class Trainer_ProformerUNC(BaseTrainer_Proformer):
    """
    Trainer class
    """
    def __init__(self, model_G0, model_G1, model_G2, model_G3, model_G4, model_G5, model_G6,
                 data_loader, valid_data_loader, criterion, metric_ftns, config,
                 optimizer_G0, optimizer_G1, optimizer_G2, optimizer_G3,optimizer_G4,optimizer_G5, optimizer_G6,
                 device,
                 lr_scheduler_G=None, len_epoch=None):
        super().__init__(
            model_G0, model_G1, model_G2, model_G3, model_G4, model_G5, model_G6,
            criterion, metric_ftns,
            optimizer_G0, optimizer_G1, optimizer_G2, optimizer_G3,optimizer_G4,optimizer_G5, optimizer_G6,config
        )
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if criterion == None:
            if self.config['trainer']['criterion']=='GEV':
                self.criterion_name = 'GEV'
                self.criterion = bayeGen_loss
        else:
            self.criterion_name = 'MSE'
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

        self.sec = self.config['data_loader']['args']['sec']
        self.batch_size = self.config['data_loader']['args']['batch_size']

        self.sec_intervals = self.config['trainer']['sec_intervals']
        self.prediction_times = self.config['trainer']['prediction_times']
        self.initial_label_len = self.config['data_loader']['args']['label_len']
        self.pred_len = self.config['data_loader']['args']['pred_len']
        self.initial_seq_len = self.config['data_loader']['args']['seq_len']

        self.use_amp = self.config['trainer']['use_amp']
        self.output_attention = self.config['arch_G6']['args']['output_attention']
        self.gamma = self.config['trainer']['gamma']
        self.lambda_steps = self.config['trainer']['lambda_steps']

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        ## Plotting for whole day
        self.multiplier_5min = int(5*60/self.sec)

        end_time = self.pred_len+self.config['data_loader']['args']['seq_len']
        if self.sec*end_time==23400:
            self.plot_whole_day_with_5min_int = True


        self.train_metrics = MetricTracker(
            str(self.criterion_name)+'loss_whole',
            str(self.criterion_name)+'loss0',str(self.criterion_name)+'loss1',str(self.criterion_name)+'loss2',str(self.criterion_name)+'loss3',str(self.criterion_name)+'loss4',str(self.criterion_name)+'loss5',str(self.criterion_name)+'loss6',
            *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            str(self.criterion_name)+'loss_whole',
            str(self.criterion_name)+'loss0',str(self.criterion_name)+'loss1',str(self.criterion_name)+'loss2',str(self.criterion_name)+'loss3',str(self.criterion_name)+'loss4',str(self.criterion_name)+'loss5',str(self.criterion_name)+'loss6', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.optimizers = [self.optimizer_G0,self.optimizer_G1,self.optimizer_G2,self.optimizer_G3,self.optimizer_G4,self.optimizer_G5,self.optimizer_G6]
 
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G0.train()
        self.model_G1.train()
        self.model_G2.train()
        self.model_G3.train()
        self.model_G4.train()
        self.model_G5.train()
        self.model_G6.train()
        
        
        train_len = len(self.data_loader) 
        # valid_len = len(self.valid_data_loader)
        prediction_steps = len(self.prediction_times)-1
        self.train_metrics.reset()
        self.MSEloss_whole = []
        self.MSElosses_vol = np.zeros((prediction_steps, self.len_epoch))
        self.MSElosses = np.zeros((prediction_steps, self.len_epoch))

        self.writer.set_step(self.tensor_step)
        
        ### Preallocate data for plotting later
        plotting_data = np.zeros((prediction_steps,self.batch_size,23400))
    
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark,stat1,stat2,stat5,stat10,stat30,stat60) in enumerate(self.data_loader):
            
            stats = [stat1,stat2,stat5,stat10,stat30,stat60]
            batch_size = batch_x.shape[0]

            batch_x = batch_x.to(torch.float32).to(self.device)
            batch_y = batch_y.to(torch.float32).to(self.device)
            batch_x_mark = batch_x_mark.to(torch.float32).to(self.device)  
            batch_y_mark = batch_y_mark.to(torch.float32).to(self.device)  
            # print('original batch_y_mark : ',batch_y_mark.shape)
            
            loss_whole=0
            loss0=0
            loss1=0
            loss2=0
            loss3=0
            loss4=0
            loss5=0
            loss6=0

            MSElosses_step = [loss0, loss1, loss2, loss3, loss4, loss5, loss6]
            
            prev_sec = self.sec
            for step in range(1,prediction_steps+1):
                # print('step-1 : ', step-1)
                model = self.models[step-1]
                
                lambda_step = self.lambda_steps[step-1]
                optimizer = self.optimizers[step-1]

                optimizer.zero_grad()
                #########################################################
                # Step 1
                #########################################################
                            
                # step = 1
                
                sec_step = self.sec_intervals[step-1]                                   # 5             # 5             # 10
                multiplier = int(sec_step/prev_sec)                                     # 1             # 1             # 2

                step_prediction_end = int(self.prediction_times[step]/sec_step)         # 360/5 = 72    # 720/5 = 144   # 1440/10=144
                step_seq_len = int(self.prediction_times[step-1]/sec_step)              # 240/5 = 48    # 360/5 = 72    # 720/10 = 72
                step_pred_len = step_prediction_end-step_seq_len                        # 72-48 = 24    # 144-72 = 72   # 144-72 = 72
                step_label_len = int(step_seq_len*2/3)                                  # 48*2/3 = 32   # 72*2/3 = 48   # 72*2/3 = 48
                # print('sec_step : ', sec_step)
                # print('step_prediction_end : ', step_prediction_end)
                # print('step_seq_len : ', step_seq_len)
                # print('step_label_len : ', step_label_len)
                if step==1:
                    batch_x1 = batch_x[:,:self.initial_seq_len,:]
                    batch_x_mark1 = batch_x_mark[:,:self.initial_seq_len,:]
                    batch_zeros = torch.zeros((batch_x1.shape[0],self.initial_seq_len,1))
                    batch_x2 = torch.cat((torch.unsqueeze(batch_x1[:,:,0],-1),batch_zeros),0)
                else:
                    batch_y = torch.unsqueeze(batch_y[:,:,0],-1)
                    
                    batch_x2 = torch.cat((batch_x2, outputs), 1)
                    # print('batch_x1.shape : ', batch_x1.shape)
                    # unnormalize, make interval to sec_step, and normalize again
                    if prev_sec!=sec_step:
                        # unnormalize volumes of (prev_sec = 5 or 10 or 30, or 60) sec interval
                        if prev_sec==1:
                            sec_index=0
                        if prev_sec==2:
                            sec_index=1
                        if prev_sec==5:
                            sec_index=2
                        if prev_sec==10:
                            sec_index=3
                        if prev_sec==30:
                            sec_index=4
                        (mean_prev, std_prev) = stats[sec_index]
                        (mean_step, std_step) = stats[sec_index+1]
                        mean_prev, std_prev = mean_prev.to(self.device), std_prev.to(self.device)
                        mean_step, std_step = mean_step.to(self.device), std_step.to(self.device)

                        batch_x2 = unnormalize_then_normalize(batch_x2,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step)

                        batch_y = unnormalize_then_normalize(batch_y,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step)
                        batch_y_mark = unnormalize_then_normalize(batch_y_mark,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step)
                        # print('batch_y.shape : ', batch_y.shape)
                    batch_x_mark1 = batch_y_mark[:,:step_seq_len,:]
                    # print('batch_x_mark1.shape : ', batch_x_mark1.shape)

                batch_y1 = batch_y[:,(step_seq_len-step_label_len):step_prediction_end,:]               # 48-32:72
                # print('batch_y1.shape : ', batch_y1.shape)
                batch_y_mark1 = batch_y_mark[:,(step_seq_len-step_label_len):step_prediction_end,:]
                # print('batch_y_mark1.shape : ', batch_y_mark1.shape)
                ### Step forward
                
                
                outputs, vol_predicted, vol_real, loss_vol = self.step_forward(step, batch_x2, batch_x_mark1, batch_y1, batch_y_mark1, step_seq_len, step_label_len, step_pred_len, model)
                
                if batch_idx==self.len_epoch-1:
                    name = 'Generated_Step_'+str(step-1)
                    plotter = torch.cat((batch_x1, outputs), 1)
                    plotter = plotter[:,:,0]
                    self.writer.add_figure(name, plot_classes_preds_with_truth(torch.squeeze(plotter).cpu().detach().numpy(),torch.squeeze(batch_y[:,:step_prediction_end,0]).cpu().detach().numpy() ))
                
                losses_step[step-1] = loss_vol
                # MSElosses_step[step-1] = MSEloss_vol1 + self.gamma*MSEloss_lob1
                # print(MSElosses_step)
                # print()

                # self.MSElosses_vol[step-1,batch_idx] = MSEloss_vol1.item()
                self.losses[step-1,batch_idx] = losses_step[step-1].item()
                
                
                # if self.use_amp:
                #     self.scaler.scale(MSElosses_step[step-1]).backward(retain_graph=True)
                #     self.scaler.step(optimizer)
                #     self.scaler.update()
                # else:
                #     MSElosses_step[step-1].backward(retain_graph=True)
                #     optimizer.step()
                #     optimizer.zero_grad()

                loss_whole += lambda_step * losses_step[step-1]
                prev_sec = sec_step

            self.loss_whole.append(loss_whole.item())
            
            ### Backwards with the whole mse loss
            if self.use_amp:
                self.scaler.scale(loss_whole).backward()
                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                loss_whole.backward()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                    optimizer.step()
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} {} loss_whole: {:.4f}'.format(
                    epoch,
                    self.epochs,
                    self._progress(batch_idx),
                    loss_whole.item()))

            if batch_idx == self.len_epoch:
                break





        losses = np.mean(self.losses,axis=1)
        

        
        loss = np.average(self.loss_whole)
        self.train_metrics.update(str(self.criterion_name)+'loss_whole', loss)
        self.train_metrics.update(str(self.criterion_name)+'loss0', losses[0].item())
        self.train_metrics.update(str(self.criterion_name)+'loss1', losses[1].item())
        self.train_metrics.update(str(self.criterion_name)+'loss2', losses[2].item())
        self.train_metrics.update(str(self.criterion_name)+'loss3', losses[3].item())
        self.train_metrics.update(str(self.criterion_name)+'loss4', losses[4].item())
        self.train_metrics.update(str(self.criterion_name)+'loss5', losses[5].item())
        self.train_metrics.update(str(self.criterion_name)+'loss6', losses[6].item())



        

        log = self.train_metrics.result()
        whole_volume = torch.cat((batch_x2, outputs), 1)
        whole_volume = whole_volume[:,:,0]
        self.writer.add_figure('Generated_Series_whole', plot_classes_preds_with_truth(torch.squeeze(whole_volume).cpu().detach().numpy(),torch.squeeze(batch_y[:,:,0]).cpu().detach().numpy() ))
        
        # outputs = outputs[:,:,0]
        # batch_y = batch_y[:,:,0]
        if self.plot_whole_day_with_5min_int:
            outputs_5min = torch.sum(whole_volume.reshape(-1,5),axis=1).reshape(whole_volume.shape[0],-1)
            batch_y_5min = torch.sum(batch_y[:,:,0].reshape(-1,5),axis=1).reshape(batch_y[:,:,0].shape[0],-1)
            self.writer.add_figure('Generated_Series_in5min', plot_classes_preds_with_truth(torch.squeeze(outputs_5min).cpu().detach().numpy(),torch.squeeze(batch_y_5min).cpu().detach().numpy() ))
            
        
        
        
        
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
        self.model_G0.eval()
        self.model_G1.eval()
        self.model_G2.eval()
        self.model_G3.eval()
        self.model_G4.eval()
        self.model_G5.eval()
        self.model_G6.eval()
        self.valid_metrics.reset()
        self.writer.set_step(self.tensor_step, 'valid')
        prediction_steps = len(self.prediction_times)-1

        valid_loss = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark,stat1,stat2,stat5,stat10,stat30,stat60) in enumerate(self.valid_data_loader):
                
                stats = [stat1,stat2,stat5,stat10,stat30,stat60]
                batch_size = batch_x.shape[0]

                batch_x = batch_x.to(torch.float32).to(self.device)
                batch_y = batch_y.to(torch.float32).to(self.device)
                batch_x_mark = batch_x_mark.to(torch.float32).to(self.device)  
                batch_y_mark = batch_y_mark.to(torch.float32).to(self.device)  

                loss_whole=0
                loss0=0
                loss1=0
                loss2=0
                loss3=0
                loss4=0
                loss5=0
                loss6=0

                losses_step = [loss0, loss1, loss2, loss3, loss4, loss5, loss6]
                
                prev_sec = self.sec
                for step in range(1,prediction_steps+1):
                    # print('step-1 : ', step-1)
                    model = self.models[step-1]
                    
                    lambda_step = self.lambda_steps[step-1]
                    
                    sec_step = self.sec_intervals[step-1]                                   # 5             # 5             # 10
                    multiplier = int(sec_step/prev_sec)                                     # 1             # 1             # 2

                    step_prediction_end = int(self.prediction_times[step]/sec_step)         # 360/5 = 72    # 720/5 = 144   # 1440/10=144
                    step_seq_len = int(self.prediction_times[step-1]/sec_step)              # 240/5 = 48    # 360/5 = 72    # 720/10 = 72
                    step_pred_len = step_prediction_end-step_seq_len                        # 72-48 = 24    # 144-72 = 72   # 144-72 = 72
                    step_label_len = int(step_seq_len*2/3)                                  # 48*2/3 = 32   # 72*2/3 = 48   # 72*2/3 = 48
                    # print('sec_step : ', sec_step)
                    # print('step_prediction_end : ', step_prediction_end)
                    # print('step_seq_len : ', step_seq_len)
                    # print('step_label_len : ', step_label_len)
                    if step==1:
                        batch_x1 = batch_x[:,:self.initial_seq_len,:]
                        batch_x_mark1 = batch_x_mark[:,:self.initial_seq_len,:]
                        batch_zeros = torch.zeros((batch_x1.shape[0],self.initial_seq_len,1))
                        batch_x2 = torch.cat((torch.unsqueeze(batch_x1[:,:,0],-1),batch_zeros),0)
                    else:
                        batch_y = torch.unsqueeze(batch_y[:,:,0],-1)
                        
                        batch_x2 = torch.cat((batch_x2, outputs), 1)
                        # print('batch_x1.shape : ', batch_x1.shape)
                        # unnormalize, make interval to sec_step, and normalize again
                        if prev_sec!=sec_step:
                            # unnormalize volumes of (prev_sec = 5 or 10 or 30, or 60) sec interval
                            if prev_sec==1:
                                sec_index=0
                            if prev_sec==2:
                                sec_index=1
                            if prev_sec==5:
                                sec_index=2
                            if prev_sec==10:
                                sec_index=3
                            if prev_sec==30:
                                sec_index=4
                            (mean_prev, std_prev) = stats[sec_index]
                            (mean_step, std_step) = stats[sec_index+1]
                            mean_prev, std_prev = mean_prev.to(self.device), std_prev.to(self.device)
                            mean_step, std_step = mean_step.to(self.device), std_step.to(self.device)

                            batch_x2 = unnormalize_then_normalize(batch_x2,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step)

                            batch_y = unnormalize_then_normalize(batch_y,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step)
                            batch_y_mark = unnormalize_then_normalize(batch_y_mark,mean_prev,std_prev,mean_step,std_step, prev_sec, sec_step)
                            # print('batch_y.shape : ', batch_y.shape)
                        batch_x_mark1 = batch_y_mark[:,:step_seq_len,:]
                        # print('batch_x_mark1.shape : ', batch_x_mark1.shape)

                    batch_y1 = batch_y[:,(step_seq_len-step_label_len):step_prediction_end,:]               # 48-32:72
                    # print('batch_y1.shape : ', batch_y1.shape)
                    batch_y_mark1 = batch_y_mark[:,(step_seq_len-step_label_len):step_prediction_end,:]
                    # print('batch_y_mark1.shape : ', batch_y_mark1.shape)
                    ### Step forward
                    
                    
                    outputs, vol_predicted, vol_real, loss_vol = self.step_forward(step, batch_x2, batch_x_mark1, batch_y1, batch_y_mark1, step_seq_len, step_label_len, step_pred_len, model)
                    

                    if batch_idx==self.len_epoch-1:
                        name = 'Generated_Step_'+str(step-1)
                        plotter = torch.cat((batch_x1, outputs), 1)
                        plotter = plotter[:,:,0]
                        self.writer.add_figure(name, plot_classes_preds_with_truth(torch.squeeze(plotter).cpu().detach().numpy(),torch.squeeze(batch_y[:,:step_prediction_end,0]).cpu().detach().numpy() ))
                    losses_step[step-1] = loss_vol
                    # MSElosses_step[step-1] = MSEloss_vol1 + self.gamma*MSEloss_lob1
                    # print(MSElosses_step)
                    # print()

                    # self.MSElosses_vol[step-1,batch_idx] = MSEloss_vol1.item()
                    self.losses[step-1,batch_idx] = losses_step[step-1].item()
                    
                    
                    # if self.use_amp:
                    #     self.scaler.scale(MSElosses_step[step-1]).backward(retain_graph=True)
                    #     self.scaler.step(optimizer)
                    #     self.scaler.update()
                    # else:
                    #     MSElosses_step[step-1].backward(retain_graph=True)
                    #     optimizer.step()
                    #     optimizer.zero_grad()

                    loss_whole += lambda_step * losses_step[step-1]
                    prev_sec = sec_step

                self.loss_whole.append(loss_whole.item())
            
            losses = np.mean(self.losses,axis=1)
            loss = np.average(self.loss_whole)
                
        
        
        self.valid_metrics.update(str(self.criterion_name)+'loss_whole', loss)
        self.valid_metrics.update(str(self.criterion_name)+'loss0', losses[0].item())
        self.valid_metrics.update(str(self.criterion_name)+'loss1', losses[1].item())
        self.valid_metrics.update(str(self.criterion_name)+'loss2', losses[2].item())
        self.valid_metrics.update(str(self.criterion_name)+'loss3', losses[3].item())
        self.valid_metrics.update(str(self.criterion_name)+'loss4', losses[4].item())
        self.valid_metrics.update(str(self.criterion_name)+'loss5', losses[5].item())
        self.valid_metrics.update(str(self.criterion_name)+'loss6', losses[6].item())

        
        whole_volume = torch.cat((batch_x2, outputs), 1)
        whole_volume = whole_volume[:,:,0]

        self.writer.add_figure('Generated_Series_valid', plot_classes_preds_with_truth(torch.squeeze(whole_volume).cpu().detach().numpy(),torch.squeeze(batch_y[:,:,0]).cpu().detach().numpy() ))

        # # print(outputs.shape)
        if self.plot_whole_day_with_5min_int:
            outputs_5min = torch.sum(whole_volume.reshape(-1,5),axis=1).reshape(whole_volume.shape[0],-1)
            batch_y_5min = torch.sum(batch_y[:,:,0].reshape(-1,5),axis=1).reshape(batch_y[:,:,0].shape[0],-1)
            self.writer.add_figure('Generated_Series_in5min_valid', plot_classes_preds_with_truth(torch.squeeze(outputs_5min).cpu().detach().numpy(),torch.squeeze(batch_y_5min).cpu().detach().numpy() ))

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model_G.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def step_forward(
        self,
        step,
        batch_x,
        batch_x_mark,
        batch_y,
        batch_y_mark,
        step_seq_len,           # 48
        step_label_len,         # 32
        step_pred_len,          # 24
        model
    ):

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -step_pred_len:, :]).to(torch.float32)
        dec_inp = torch.cat([batch_y[:, :step_label_len, :], dec_inp], dim=1).to(torch.float32).to(self.device)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if self.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,step_label_len,step_pred_len)[:-1]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,step_label_len,step_pred_len)

                # outputs = outputs[:, -step_pred_len:,:]
                batch_y = batch_y[:, -step_pred_len:,:].to(self.device)

                vol_predicted = outputs[0][:,-step_pred_len:,0]
                vol_real = batch_y[:,-step_pred_len:,0]

                # lob_predicted = outputs[:,:,1:]
                # lob_real = batch_y[:,:,1:]
                # if len(outputs.shape)==3:
                #     outputs = torch.squeeze(outputs)
                loss_vol = self.criterion(outputs[0][:,-step_pred_len:,0],outputs[1][:,-step_pred_len:,0],outputs[2][:,-step_pred_len:,0], vol_real)
                # loss_vol = self.criterion(vol_predicted, vol_real)
                # MSEloss_lob = self.criterion(lob_predicted, lob_real)
                # self.train_loss.append(MSEloss.item())            #######important
                
        else:
            if self.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,step_label_len,step_pred_len)[:-1]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,step_label_len,step_pred_len)
            # print('outputs.shape : ', outputs.shape)
            # outputs = outputs[:, -step_pred_len:,:]
            batch_y = batch_y[:, -step_pred_len:,:].to(self.device)

            # print('######### outputs.shape: ', outputs.shape)
            # print('######### batch_y.shape: ', batch_y.shape)

            vol_predicted = outputs[0][:,-step_pred_len:,0]
            vol_real = batch_y[:,-step_pred_len:,0]

            # if len(outputs.shape)==3:
            #     outputs = torch.squeeze(outputs)
            loss_vol = self.criterion(outputs[0][:,-step_pred_len:,0],outputs[1][:,-step_pred_len:,0],outputs[2][:,-step_pred_len:,0], vol_real)
            # MSEloss_lob = self.criterion(lob_predicted, lob_real)

            # self.train_loss.append(MSEloss.item())

        return outputs, vol_predicted, vol_real, loss_vol



    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


