import yaml
import argparse
import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
from src.models.dataset import CranesDataset
from torch.utils.data import DataLoader

def __visualize(config_path: str, mode='train'):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    assert mode == 'train' or mode == 'valid' or mode == 'test'

    if mode == 'train': 
        frames_path = config['split_dataset']['train_frames_path']
        masks_path = config['split_dataset']['train_masks_path']
    elif mode == 'test':
        frames_path = config['split_dataset']['test_frames_path']
        masks_path = config['split_dataset']['test_masks_path']
    else:
        frames_path = config['split_dataset']['valid_frames_path']
        masks_path = config['split_dataset']['valid_masks_path']

    dest_height = config['train_model']['dest_height']
    dest_width = config['train_model']['dest_width']
    dataset = CranesDataset(frames_path, masks_path, dest_height, dest_width)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data, target in loader:
        data = data.reshape(data.shape[1], data.shape[2], data.shape[3])
        data *= 255.0
        data = data.type(torch.uint8)

        target = target.bool()

        out = draw_segmentation_masks(data, target,
            alpha=0.4,
            colors=(255, 0, 0)
        )
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 0, 1)

        plt.imshow(out) #needs (H, W, 3) - current (3, H, W)
        plt.show()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--mode', dest='mode')
    args = args_parser.parse_args()

    __visualize(args.config, args.mode)