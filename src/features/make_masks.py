import cv2
import yaml
import json
import os
import numpy as np
import argparse

def __make_masks(config_path: str) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    labels_path = config['make_frames']['labels_path']
    frames_path = config['make_frames']['frames_path']
    masks_path = config['make_masks']['masks_path']

    for label in list(filter(lambda x: x.endswith('.json'), \
        os.listdir(labels_path))):

        frames_with_polygons = __get_frames_with_polygons(
            os.path.join(labels_path, label)
        )
        for k, v in frames_with_polygons.items():
            frame = cv2.imread(os.path.join(frames_path, k))
            W, H = frame.shape[1], frame.shape[0]
            blank = np.zeros((H, W))
            for object in v:
                cv2.fillPoly(blank, np.array([object]), (255, 255, 255))
                cv2.imwrite(os.path.join(masks_path, k), blank)

def __get_frames_with_polygons(label_path: str) -> dict:
    """
        args:
            label_path: str - path to label json file, with indicies
                that have corresponding image e.g. h0_00-02.mp4.json,
                h0_00-02_0.png ...
        returns:
            frames_with_polygons: dict - dictionary with frame path
                as key and list of polygons as value 
    """
    
    with open(label_path, 'r') as json_file:
        label = json.load(json_file)
    
    frames_with_polygons = {}
    for n, idx in enumerate(label['frames']):
        frame_name = __get_frame_name(label_path, n)
        for fig in idx['figures']:
            if frame_name not in frames_with_polygons:
                frames_with_polygons[frame_name] = \
                    [fig['geometry']['points']['exterior']]
            else:
                frames_with_polygons[frame_name].append(
                    fig['geometry']['points']['exterior']
                )

    return frames_with_polygons

def __get_frame_name(label_path: str, idx: int) -> str:
    frame_name = label_path.split('/')[-1]
    frame_name = frame_name.split('.')[0]
    frame_name = frame_name + '_' + str(idx) + '.png'
    return frame_name

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    __make_masks(args.config)