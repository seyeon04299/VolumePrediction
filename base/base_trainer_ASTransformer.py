import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer_ASTransformer:
    """
    Base class for ASTransformer
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model_G = model_G
        self.model_D = model_D
        
        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
    
        cfg_trainer = config['trainer']
        self.model_name = cfg_trainer['model_name']

        self.epochs = cfg_trainer['epochs']

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

        if config.resume_G is not None:
            if config.resume_D is not None:
                self._resume_checkpoint(config.resume_G, config.resume_D)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError




    def train_ASTransformer(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
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
        arch_G = type(self.model_G).__name__
        state_G = {
            'arch_G': arch_G,
            'epoch': epoch,
            'state_dict_G': self.model_G.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'G_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G_best.pth')
            torch.save(state_G, best_path)
            self.logger.info("Saving current best: model_G_best.pth ...")


        arch_D = type(self.model_D).__name__
        state_D = {
            'arch_D': arch_D,
            'epoch': epoch,
            'state_dict_D': self.model_D.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'D_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_D, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_D_best.pth')
            torch.save(state_D, best_path)
            self.logger.info("Saving current best: model_D_best.pth ...")



    def _resume_checkpoint(self, resume_path_G, resume_path_D):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path_G = str(resume_path_G)
        resume_path_D = str(resume_path_D)
        
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_D))

        checkpoint_G = torch.load(resume_path_G)
        self.start_epoch = checkpoint_G['epoch'] + 1
        self.mnt_best = checkpoint_G['monitor_best']

        checkpoint_D = torch.load(resume_path_D)
        self.start_epoch = checkpoint_D['epoch'] + 1
        self.mnt_best = checkpoint_D['monitor_best']

        # load architecture params from checkpoint. (Generator)
        if checkpoint_G['config']['arch_G'] != self.config['arch_G']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G.load_state_dict(checkpoint_G['state_dict'])

        # load architecture params from checkpoint. (Discriminator)
        if checkpoint_D['config']['arch_D'] != self.config['arch_D']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_D.load_state_dict(checkpoint_D['state_dict'])


        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint_G['config']['optimizer_G']['type'] != self.config['optimizer_G']['type']:
            self.logger.warning("Warning: Optimizer_G type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G.load_state_dict(checkpoint_G['optimizer_G'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint_D['config']['optimizer_D']['type'] != self.config['optimizer_D']['type']:
            self.logger.warning("Warning: Optimizer_D type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_D.load_state_dict(checkpoint_D['optimizer_D'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
