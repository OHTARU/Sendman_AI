import os
import time
import torch
import torch.nn as nn

from kospeech.utils import logger
from kospeech.data import SpectrogramDataset
from kospeech.models import ListenAttendSpell
from kospeech.optim import Optimizer


class Checkpoint(object):
    SAVE_PATH = 'outputs'
    LOAD_PATH = '../../../outputs'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(
            self,
            model: nn.Module = None,                   
            optimizer: Optimizer = None,               
            trainset_list: list = None,                
            validset: SpectrogramDataset = None,       
            epoch: int = None,                         
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.trainset_list = trainset_list
        self.validset = validset
        self.epoch = epoch

    def save(self):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        checkpoint_dir = os.path.join(self.SAVE_PATH, date_time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        trainer_states = {
            'optimizer': self.optimizer,
            'trainset_list': self.trainset_list,
            'validset': self.validset,
            'epoch': self.epoch,
        }
        torch.save(trainer_states, os.path.join(checkpoint_dir, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(checkpoint_dir, self.MODEL_NAME))
        logger.info('save checkpoints\n%s\n%s'
                    % (os.path.join(checkpoint_dir, self.TRAINER_STATE_NAME),
                       os.path.join(checkpoint_dir, self.MODEL_NAME)))

    def load(self, path):
        logger.info('load checkpoints\n%s\n%s'
                    % (os.path.join(path, self.TRAINER_STATE_NAME),
                       os.path.join(path, self.MODEL_NAME)))

        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, self.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, self.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, self.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, self.MODEL_NAME), map_location=lambda storage, loc: storage)

        if isinstance(model, ListenAttendSpell):
            if isinstance(model, nn.DataParallel):
                model.module.flatten_parameters()  
            else:
                model.flatten_parameters()

        return Checkpoint(
            model=model, 
            optimizer=resume_checkpoint['optimizer'], 
            epoch=resume_checkpoint['epoch'],
            trainset_list=resume_checkpoint['trainset_list'],
            validset=resume_checkpoint['validset'],
        )

    def get_latest_checkpoint(self):
        if not os.path.isdir(self.LOAD_PATH):
            os.makedirs(self.LOAD_PATH)
            return None

        checkpoints_dirs = [d for d in os.listdir(self.LOAD_PATH) if os.path.isdir(os.path.join(self.LOAD_PATH, d))]
        if not checkpoints_dirs:
            return None

        latest_checkpoint_dir = sorted(checkpoints_dirs, reverse=True)[0]
        checkpoint_path = os.path.join(self.LOAD_PATH, latest_checkpoint_dir)

        trainer_state_path = os.path.join(checkpoint_path, self.TRAINER_STATE_NAME)
        model_path = os.path.join(checkpoint_path, self.MODEL_NAME)
        if not os.path.exists(trainer_state_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint files not found in {checkpoint_path}.")

        return checkpoint_path

# 예시 사용법:
checkpoint = Checkpoint()
latest_checkpoint_path = checkpoint.get_latest_checkpoint()

if latest_checkpoint_path:
    print(f"Latest checkpoint path: {latest_checkpoint_path}")
else:
    print("No checkpoint found. Initializing training from scratch.")
