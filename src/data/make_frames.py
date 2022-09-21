"""
    THIS CODE REQUIRES FFMPEG BINARY INSTALLED
"""

import yaml
import os
import json
import argparse

def __make_frames(config_path: str) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    videos_path = config['split_2m']['vids_2m_path']
    labels_path = config['make_frames']['labels_path']
    frames_path = config['make_frames']['frames_path']
    cmd = 'ffmpeg -i {0} -vf select="eq(n\,{1})" \
-vsync vfr -q:v 2 -vcodec png {2}'

    for label in list(filter(lambda x: x.endswith('.json'), \
        os.listdir(labels_path))):
        indicies = __get_indicies(os.path.join(labels_path, label))
        for n, idx in enumerate(indicies):
            out_file = os.path.join(frames_path, \
                label.split('.')[0]+'_'+str(n)+'.png')
            os.system(cmd.format(os.path.join(videos_path, \
                __get_video_name(label)), idx, out_file))

def __get_indicies(label_path: str) -> list:
    with open(label_path) as labels_file:
        labels = json.load(labels_file)
    
    indicies = []
    for idx in labels['frames']:
        idx = idx['index']
        indicies.append(idx)
    return indicies

def __get_video_name(label: str) -> str:
    return label.split('.')[0] + '.mp4'

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args() 

    __make_frames(args.config)