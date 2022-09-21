import zipfile
import yaml
import os
import argparse

def __extract_all(config_path: str) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    zipped_path = config['unzip_raw']['zip_path']
    hevc_path = config['unzip_raw']['hevc_path']

    for zip_file in list(filter(lambda x: x.endswith('.zip'), \
        os.listdir(zipped_path))):

        with zipfile.ZipFile(os.path.join(zipped_path, zip_file), 'r') \
            as zip_ref:
            zip_ref.extractall(hevc_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    __extract_all(args.config)