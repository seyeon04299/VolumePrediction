import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer_Proformer:
    """
    Base class for Proformer
    """
    def __init__(
        self, model_G0, model_G1, model_G2, model_G3, model_G4, model_G5, model_G6,
            criterion, metric_ftns,
            optimizer_G0, optimizer_G1, optimizer_G2, optimizer_G3,optimizer_G4,optimizer_G5, optimizer_G6,config
    ):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model_G0 = model_G0
        self.model_G1 = model_G1
        self.model_G2 = model_G2
        self.model_G3 = model_G3
        self.model_G4 = model_G4
        self.model_G5 = model_G5
        self.model_G6 = model_G6
        self.models = [model_G0,model_G1,model_G2,model_G3,model_G4,model_G5,model_G6]
        
        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.optimizer_G0 = optimizer_G0
        self.optimizer_G1 = optimizer_G1
        self.optimizer_G2 = optimizer_G2
        self.optimizer_G3 = optimizer_G3
        self.optimizer_G4 = optimizer_G4
        self.optimizer_G5 = optimizer_G5
        self.optimizer_G6 = optimizer_G6
    
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

        if config.resume_G0 is not None:
            if config.resume_G1 is not None:
                self._resume_checkpoint(config.resume_G0, config.resume_G1, config.resume_G2, config.resume_G3, config.resume_G4, config.resume_G5, config.resume_G6)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError




    def train_Proformer(self):
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
        arch_G0 = type(self.model_G0).__name__
        state_G0 = {
            'arch_G0': arch_G0,
            'epoch': epoch,
            'state_dict_G': self.model_G0.state_dict(),
            'optimizer_G': self.optimizer_G0.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename0 = str(self.checkpoint_dir / 'G0_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G0, filename0)
        self.logger.info("Saving checkpoint: {} ...".format(filename0))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G0_best.pth')
            torch.save(state_G0, best_path)
            self.logger.info("Saving current best: model_G0_best.pth ...")


        arch_G1 = type(self.model_G1).__name__
        state_G1 = {
            'arch_G': arch_G1,
            'epoch': epoch,
            'state_dict_G1': self.model_G1.state_dict(),
            'optimizer_G1': self.optimizer_G1.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename1 = str(self.checkpoint_dir / 'G1_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G1, filename1)
        self.logger.info("Saving checkpoint: {} ...".format(filename1))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G1_best.pth')
            torch.save(state_G1, best_path)
            self.logger.info("Saving current best: model_G1_best.pth ...")


        arch_G2 = type(self.model_G2).__name__
        state_G2 = {
            'arch_G2': arch_G2,
            'epoch': epoch,
            'state_dict_G2': self.model_G2.state_dict(),
            'optimizer_G2': self.optimizer_G2.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename2 = str(self.checkpoint_dir / 'G2_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G2, filename2)
        self.logger.info("Saving checkpoint: {} ...".format(filename2))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G2_best.pth')
            torch.save(state_G2, best_path)
            self.logger.info("Saving current best: model_G2_best.pth ...")


        arch_G3 = type(self.model_G3).__name__
        state_G3 = {
            'arch_G3': arch_G3,
            'epoch': epoch,
            'state_dict_G3': self.model_G3.state_dict(),
            'optimizer_G3': self.optimizer_G3.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename3 = str(self.checkpoint_dir / 'G3_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G3, filename3)
        self.logger.info("Saving checkpoint: {} ...".format(filename3))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G3_best.pth')
            torch.save(state_G3, best_path)
            self.logger.info("Saving current best: model_G3_best.pth ...")


        arch_G4 = type(self.model_G4).__name__
        state_G4 = {
            'arch_G4': arch_G4,
            'epoch': epoch,
            'state_dict_G4': self.model_G4.state_dict(),
            'optimizer_G4': self.optimizer_G4.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename4 = str(self.checkpoint_dir / 'G4_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G4, filename4)
        self.logger.info("Saving checkpoint: {} ...".format(filename4))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G4_best.pth')
            torch.save(state_G4, best_path)
            self.logger.info("Saving current best: model_G4_best.pth ...")


        arch_G5 = type(self.model_G5).__name__
        state_G5 = {
            'arch_G5': arch_G5,
            'epoch': epoch,
            'state_dict_G5': self.model_G5.state_dict(),
            'optimizer_G5': self.optimizer_G5.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename5 = str(self.checkpoint_dir / 'G5_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G5, filename5)
        self.logger.info("Saving checkpoint: {} ...".format(filename5))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G5_best.pth')
            torch.save(state_G5, best_path)
            self.logger.info("Saving current best: model_G5_best.pth ...")


        arch_G6 = type(self.model_G6).__name__
        state_G6 = {
            'arch_G6': arch_G6,
            'epoch': epoch,
            'state_dict_G6': self.model_G6.state_dict(),
            'optimizer_G6': self.optimizer_G6.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename6 = str(self.checkpoint_dir / 'G6_checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state_G6, filename6)
        self.logger.info("Saving checkpoint: {} ...".format(filename6))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_G6_best.pth')
            torch.save(state_G6, best_path)
            self.logger.info("Saving current best: model_G6_best.pth ...")




    def _resume_checkpoint(self, resume_path_G0, resume_path_G1,resume_path_G2, resume_path_G3, resume_path_G4, resume_path_G5, resume_path_G6):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path_G0 = str(resume_path_G0)
        resume_path_G1 = str(resume_path_G1)
        resume_path_G2 = str(resume_path_G2)
        resume_path_G3 = str(resume_path_G3)
        resume_path_G4 = str(resume_path_G4)
        resume_path_G5 = str(resume_path_G5)
        resume_path_G6 = str(resume_path_G6)
        # resume_path_D = str(resume_path_D)
        
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G0))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G1))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G2))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G3))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G4))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G5))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path_G6))
        
        # self.logger.info("Loading checkpoint: {} ...".format(resume_path_D))

        checkpoint_G0 = torch.load(resume_path_G0)
        checkpoint_G1 = torch.load(resume_path_G1)
        checkpoint_G2 = torch.load(resume_path_G2)
        checkpoint_G3 = torch.load(resume_path_G3)
        checkpoint_G4 = torch.load(resume_path_G4)
        checkpoint_G5 = torch.load(resume_path_G5)
        checkpoint_G6 = torch.load(resume_path_G6)
        


        self.start_epoch = checkpoint_G0['epoch'] + 1
        self.mnt_best = checkpoint_G0['monitor_best']

        # checkpoint_D = torch.load(resume_path_D)
        # self.start_epoch = checkpoint_D['epoch'] + 1
        # self.mnt_best = checkpoint_D['monitor_best']

        # load architecture params from checkpoint. (Generator)
        if checkpoint_G0['config']['arch_G0'] != self.config['arch_G0']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G0.load_state_dict(checkpoint_G0['state_dict'])

        if checkpoint_G1['config']['arch_G1'] != self.config['arch_G1']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G1.load_state_dict(checkpoint_G1['state_dict'])

        if checkpoint_G2['config']['arch_G2'] != self.config['arch_G2']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G2.load_state_dict(checkpoint_G2['state_dict'])

        if checkpoint_G3['config']['arch_G3'] != self.config['arch_G3']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G3.load_state_dict(checkpoint_G3['state_dict'])

        if checkpoint_G4['config']['arch_G4'] != self.config['arch_G4']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G4.load_state_dict(checkpoint_G4['state_dict'])

        if checkpoint_G5['config']['arch_G5'] != self.config['arch_G5']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G5.load_state_dict(checkpoint_G5['state_dict'])

        if checkpoint_G6['config']['arch_G6'] != self.config['arch_G6']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G6.load_state_dict(checkpoint_G6['state_dict'])


        # # load architecture params from checkpoint. (Discriminator)
        # if checkpoint_D['config']['arch_D'] != self.config['arch_D']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")
        # self.model_D.load_state_dict(checkpoint_D['state_dict'])


        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint_G0['config']['optimizer_G0']['type'] != self.config['optimizer_G0']['type']:
            self.logger.warning("Warning: Optimizer_G0 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G0.load_state_dict(checkpoint_G0['optimizer_G0'])

        if checkpoint_G1['config']['optimizer_G1']['type'] != self.config['optimizer_G1']['type']:
            self.logger.warning("Warning: Optimizer_G1 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G1.load_state_dict(checkpoint_G1['optimizer_G1'])

        if checkpoint_G2['config']['optimizer_G2']['type'] != self.config['optimizer_G2']['type']:
            self.logger.warning("Warning: Optimizer_G2 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G2.load_state_dict(checkpoint_G2['optimizer_G2'])

        if checkpoint_G3['config']['optimizer_G3']['type'] != self.config['optimizer_G3']['type']:
            self.logger.warning("Warning: Optimizer_G3 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G3.load_state_dict(checkpoint_G3['optimizer_G3'])

        if checkpoint_G4['config']['optimizer_G4']['type'] != self.config['optimizer_G4']['type']:
            self.logger.warning("Warning: Optimizer_G4 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G4.load_state_dict(checkpoint_G4['optimizer_G4'])

        if checkpoint_G5['config']['optimizer_G5']['type'] != self.config['optimizer_G5']['type']:
            self.logger.warning("Warning: Optimizer_G5 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G5.load_state_dict(checkpoint_G5['optimizer_G5'])

        if checkpoint_G6['config']['optimizer_G6']['type'] != self.config['optimizer_G6']['type']:
            self.logger.warning("Warning: Optimizer_G6 type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G6.load_state_dict(checkpoint_G6['optimizer_G6'])

        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint_D['config']['optimizer_D']['type'] != self.config['optimizer_D']['type']:
        #     self.logger.warning("Warning: Optimizer_D type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        # else:
        #     self.optimizer_D.load_state_dict(checkpoint_D['optimizer_D'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
