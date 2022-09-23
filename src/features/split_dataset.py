import yaml
import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def __split(config_path: str) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    frames_path = config['make_frames']['frames_path']
    masks_path = config['make_masks']['masks_path']
    test_frames_path = config['split_dataset']['test_frames_path']
    test_masks_path = config['split_dataset']['test_masks_path']
    train_frames_path = config['split_dataset']['train_frames_path']
    train_masks_path = config['split_dataset']['train_masks_path']
    valid_frames_path = config['split_dataset']['valid_frames_path']
    valid_masks_path = config['split_dataset']['valid_masks_path']

    valid_perc = config['split_dataset']['valid_perc']
    test_perc = config['split_dataset']['test_perc']

    all_masks = list(filter(lambda x: x.endswith('.png'), \
        os.listdir(masks_path)))
    all_frames = list(filter(lambda x: x in all_masks, \
        os.listdir(frames_path)))
    
    frames_train, frames_test, masks_train, masks_test = \
        train_test_split(all_frames, all_masks,
            test_size=test_perc,
            shuffle=True,
            random_state=42
        )
    frames_train, frames_valid, masks_train, masks_valid = \
        train_test_split(frames_train, masks_train,
            test_size=valid_perc,
            shuffle=True,
            random_state=42
        )
    __copy_set(frames_train, frames_path, train_frames_path)
    __copy_set(masks_train, masks_path, train_masks_path)
    __copy_set(frames_valid, frames_path, valid_frames_path)
    __copy_set(masks_valid, masks_path, valid_masks_path)
    __copy_set(frames_test, frames_path, test_frames_path)
    __copy_set(masks_test, masks_path, test_masks_path)

def __copy_set(set: list, from_path: str, dest_path: str) -> None:
    for data in set:
        shutil.copy(os.path.join(from_path, data), \
            os.path.join(dest_path, data))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    __split(args.config)