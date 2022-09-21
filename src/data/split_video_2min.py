"""
    THIS CODE REQUIRES FFMPEG BINARY INSTALLED
"""
#TODO: only for below 1h vids

import yaml
import os
import cv2
import argparse

def __split2m(config_path: str) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    full_vids_path = config['change_codec']['h264_path']
    split2m_path = config['split_2m']['vids_2m_path']
    cmd = 'ffmpeg -ss 00:{0}:00 -t 00:02:00 -i {1} -acodec copy -vcodec copy {2}'
    video_extension = '.mp4'

    for full_video in list(filter(lambda x: x.endswith(video_extension), \
        os.listdir(full_vids_path))):
        full_video_path = os.path.join(full_vids_path, full_video)
        for min in range(0, __get_duration(full_video_path), 2):
            min_start = str(min) if len(str(min)) == 2 else '0' + str(min)
            min_end = str(min + 2) if len(str(min + 2)) == 2 else '0' + str(min + 2)
            out_file = os.path.join(split2m_path, \
                full_video.split('.')[0]+'_'+min_start+'-'+min_end+'.mp4')
            os.system(cmd.format(min_start, full_video_path, out_file))

def __get_duration(vid_path: str) -> int:
    cap = cv2.VideoCapture(vid_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    seconds = num_frames / FPS
    return int(seconds // 60)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args() 

    __split2m(args.config)