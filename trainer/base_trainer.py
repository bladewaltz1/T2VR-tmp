import os
from abc import abstractmethod

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard.writer import SummaryWriter

from config.base_config import Config


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config: Config, local_rank=0):
        self.config = config
        # setup GPU device if available, move model into configured device
        self.device = torch.device(config.device)

        model = model.to(self.device)
        self.model = DistributedDataParallel(model, 
                                             device_ids=[local_rank], 
                                             find_unused_parameters=True,
                                             output_device=local_rank)
        self.loss = loss.to(self.device)
        self.metrics = metrics
        self.optimizer = optimizer
        self.start_epoch = 1
        self.global_step = 0

        self.num_epochs = config.num_epochs
        self.checkpoint_dir = config.model_path

        self.log_step = config.log_step
        self.evals_per_epoch = config.evals_per_epoch

        self.local_rank = local_rank
        if self.local_rank == 0:
            self.writer = SummaryWriter(log_dir=config.tb_log_dir)
        else:
            self.writer = None

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Training logic for a step in an epoch
        :param epoch: Current epoch number
               step: Current step in epoch
               num_steps: Number of steps in epoch
        """
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            result = self._train_epoch(epoch)
            if self.local_rank == 0 and epoch % self.config.save_every == 0:
                    self._save_checkpoint(epoch, save_best=False)

    def validate(self):
        self._valid_epoch_step(0,0,0)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, save checkpoint to 'model_best.pth'
        """

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))

    def load_checkpoint(self, model_name):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
        state_dict = checkpoint['state_dict']

        self.model.load_state_dict(state_dict)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded")

