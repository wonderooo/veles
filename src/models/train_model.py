import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import ResNet101_Weights
from dataset import CranesDataset
from unet_model import UnetModel
from unet3.unet3pp import UNet_3Plus
import yaml
import argparse
import numpy as np
import utils

class TrainingSession(object):
    def __init__(self, config_path: str) -> None:
        super(TrainingSession, self).__init__()
        
        with open(config_path) as config_file:
            self.config = yaml.safe_load(config_file)

        self.lr: float = self.config['train_model']['learning_rate']        
        self.batch_size: int = self.config['train_model']['batch_size']
        self.num_epochs: int = self.config['train_model']['num_epochs']
        self.dest_height: int = self.config['train_model']['dest_height']
        self.dest_width: int = self.config['train_model']['dest_width']

        self.device = torch.device(self.config['train_model']['device'])
        self.pin_memory: bool = self.config['train_model']['pin_memory']
        self.num_workers: int = self.config['train_model']['num_workers']

        self.in_features: int = self.config['train_model']['in_features']
        self.out_features: int = self.config['train_model']['out_features']
        
        self.model_arch: str = self.config['train_model']['model_arch']
        self.load_model: bool = self.config['train_model']['load_model']
        self.checkpoint_path: str = self.config['train_model']['checkpoint_path']
        
        if self.model_arch == 'unet':
            self.model = UnetModel(self.in_features, self.out_features).to(self.device)
        elif self.model_arch == 'unet3pp':
            self.model = UNet_3Plus().to(self.device)
        elif self.model_arch == 'deeplabv3':
            self.model = deeplabv3_resnet101(num_classes=2, \
                weights_backbone=ResNet101_Weights.IMAGENET1K_V2
            ).to(self.device)
        utils.load_checkpoint(torch.load(self.checkpoint_path), self.model) \
            if self.load_model else None

        self.loss_fn = nn.BCEWithLogitsLoss() if self.model_arch != 'deeplabv3' \
                else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler()

        self.valid_frames_path = self.config['split_dataset']['valid_frames_path']
        self.valid_masks_path = self.config['split_dataset']['valid_masks_path']
        self.train_frames_path = self.config['split_dataset']['train_frames_path']
        self.train_masks_path = self.config['split_dataset']['train_masks_path']
        self.train_ds = CranesDataset(self.train_frames_path, self.train_masks_path,
            self.dest_height, self.dest_width
        )
        self.valid_ds = CranesDataset(self.valid_frames_path, self.valid_masks_path,
            self.dest_height, self.dest_width
        )
        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True
        )
        self.valid_loader = DataLoader(self.valid_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False
        )

    def __repr__(self) -> str:
        return ('Training parameters: \n' +
            f'in_features: {self.in_features} - {type(self.in_features)}\n' +
            f'out_features: {self.out_features} - {type(self.out_features)}\n' +
            f'dest_height: {self.dest_height} - {type(self.dest_height)}\n' +
            f'dest_width: {self.dest_width} - {type(self.dest_width)}\n' +
            f'batch_size: {self.batch_size} - {type(self.batch_size)}\n' +
            f'pin_memory: {self.pin_memory} - {type(self.pin_memory)}\n' +
            f'num_workers: {self.num_workers} - {type(self.num_workers)}\n' +
            f'device: {self.device} - {type(self.device)}\n' + 
            f'learning_rate: {self.lr} - {type(self.lr)}\n' +
            f'load_model: {self.load_model} - {type(self.load_model)}\n' +
            f'checkpoint_path: {self.checkpoint_path} - {type(self.checkpoint_path)}'
        )

    def __train(self) -> float:
        train_loss = 0.0
        for data, target in self.train_loader:
            self.model.train()
            data = data.to(self.device)
            if self.model_arch != 'deeplabv3':
                target = target.float().unsqueeze(1).to(self.device)
            else:
                target = target.long().to(self.device)

            """
                Forward call
            """
            with torch.cuda.amp.autocast():
                if self.model_arch != 'deeplabv3':
                    predictions = self.model(data)
                else:
                    predictions = self.model(data)['out']
                loss = self.loss_fn(predictions, target)
                print('did loss')

            """
                Backward call
            """
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item() * data.size(0)
        return train_loss
    
    def __valid(self) -> float:
        valid_loss = 0.0
        for data, target in self.valid_loader:
            self.model.eval()
            data = data.to(self.device)
            target = target.float().unsqueeze(1).to(self.device)

            """
                Forward call
            """
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                loss = self.loss_fn(predictions, target)

            valid_loss += loss.item() * data.size(0)
        return valid_loss
    
    def train(self):
        valid_min_loss = np.Inf

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.__train()
            train_loss = train_loss / len(self.train_loader.sampler)
            valid_loss = self.__valid()
            valid_loss = valid_loss / len(self.valid_loader.sampler)

            print('Epoch: {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}'
                .format(epoch, train_loss, valid_loss))

            if valid_loss <= valid_min_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                    .format(
                        valid_min_loss,
                        valid_loss
                    )
                )

                state = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                utils.save_checkpoint(state, self.checkpoint_path)

                valid_min_loss = valid_loss

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    ts = TrainingSession(args.config)
    print(ts)
    ts.train()
