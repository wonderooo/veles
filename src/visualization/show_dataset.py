import yaml
import os
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt

def __visualize(config_path: str, set_percent=1.0, mode='train'):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    assert set_percent <= 1.0 and set_percent > 0.0
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
