#TODO: remove directory on move

from genericpath import isdir
import yaml
import os
import argparse

def __clean(config_path: str) -> None:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    video_extension = '.mp4'
    hevc_path = config['unzip_raw']['hevc_path']

    for dir in os.listdir(hevc_path):
        dir_path = os.path.join(hevc_path, dir)
        if os.path.isdir(dir_path):
            vids =list(filter(lambda x: x.endswith(video_extension), \
                os.listdir(dir_path)))
            [os.rename(os.path.join(dir_path, vid), os.path.join(hevc_path, \
                vid)) for vid in vids]

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    __clean(args.config)