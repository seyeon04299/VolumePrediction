import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer_TimeGAN:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        
        self.criterion = criterion
        self.metric_ftns = metric_ftns


        self.optimizer_e = config.init_obj('optimizer_e', torch.optim, self.model.parameters())
        self.optimizer_r = config.init_obj('optimizer_r', torch.optim, self.model.parameters())
        self.optimizer_s = config.init_obj('optimizer_s', torch.optim, self.model.parameters())
        self.optimizer_g = config.init_obj('optimizer_g', torch.optim, self.model.parameters())
        self.optimizer_d = config.init_obj('optimizer_d', torch.optim, self.model.parameters())

    
        cfg_trainer = config['trainer']
        self.model_name = cfg_trainer['model_name']

        self.epochs_emb = cfg_trainer['epochs_emb']
        self.epochs_sup = cfg_trainer['epochs_sup']
        self.epochs_joint = cfg_trainer['epochs_joint']

        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')


        self.tensor_step = 0

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch_emb(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
        

    @abstractmethod
    def _train_epoch_sup(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    @abstractmethod
    def _train_epoch_joint(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError





    def train_TimeGAN(self):
        """
        Full training logic
        """
        not_improved_count = 0

        ###############################################################
        # Train Embedding 
        ###############################################################
        print("\nStart Embedding Network Training")

        for epoch in range(self.start_epoch, self.epochs_emb + 1):
            result = self._train_epoch_emb(epoch)

            # save logged informations into log dict
            log = {'epoch_emb': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch, save_best=best)

        ###############################################################
        # Train Supervisor 
        ###############################################################
        print("\nStart Training with Supervised Loss Only")

        for epoch in range(self.start_epoch, self.epochs_sup + 1):
            result = self._train_epoch_sup(epoch)

            # save logged informations into log dict
            log = {'epoch_sup': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch, save_best=best)

        ###############################################################
        # Train Joint 
        ###############################################################
        print("\nStart Joint Training")

        for epoch in range(self.start_epoch, self.epochs_joint + 1):
            result = self._train_epoch_joint(epoch)

            # save logged informations into log dict
            log = {'epoch_joint': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)





    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_e': self.optimizer_e.state_dict(),
            'optimizer_r': self.optimizer_r.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_s': self.optimizer_s.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")



    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        
        
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        

        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        
        # load architecture params from checkpoint. (Generator)
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])




        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_e']['type'] != self.config['optimizer_e']['type']:
            self.logger.warning("Warning: Optimizer_e type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_e.load_state_dict(checkpoint['optimizer_e'])
        
    
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_r']['type'] != self.config['optimizer_r']['type']:
            self.logger.warning("Warning: Optimizer_e type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_r.load_state_dict(checkpoint['optimizer_r'])
        

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_g']['type'] != self.config['optimizer_g']['type']:
            self.logger.warning("Warning: Optimizer_g type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_d']['type'] != self.config['optimizer_d']['type']:
            self.logger.warning("Warning: Optimizer_d type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
                
                
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_s']['type'] != self.config['optimizer_s']['type']:
            self.logger.warning("Warning: Optimizer_e type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_s.load_state_dict(checkpoint['optimizer_s'])
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
