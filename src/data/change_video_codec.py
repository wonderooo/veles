"""
    THIS CODE REQUIRES FFMPEG BINARY INSTALLED
    ALONG WITH LIBX264 CODEC LIBRARY
"""

import yaml
import os
import argparse

def __recodec(config_path: str) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    hevc_path = config['unzip_raw']['hevc_path']
    h264_path = config['change_codec']['h264_path']
    cmd = 'ffmpeg -i {0} -vcodec libx264 {1}'
    video_extension = '.mp4'

    for hevc in list(filter(lambda x: x.endswith(video_extension), \
        os.listdir(hevc_path))):

        in_dir = os.path.join(hevc_path, hevc)
        out_dir = os.path.join(h264_path, hevc)
        os.system(cmd.format(in_dir, out_dir))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    __recodec(args.config)
