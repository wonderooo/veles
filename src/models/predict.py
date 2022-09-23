import yaml
from unet_model import UnetModel
from utils import load_checkpoint
import torch
import argparse

class Predict(object):
    def __init__(self, config_path: str):
        with open(config_path) as config_file:
            self.config = yaml.safe_load(config_file)

        self.in_channels = self.config['train_model']['in_features'] 
        self.out_channels = self.config['train_model']['out_features']
        self.threshold = self.config['evaluate_model']['threshold']
        self.checkpoint = self.config['train_model']['checkpoint_path']
        self.device = self.config['train_model']['device']
        self.model = UnetModel(self.in_channels, self.out_channels).to(self.device)
        load_checkpoint(torch.load(self.checkpoint), self.model)

    def __call__(self, tensor: torch.Tensor):
        """
            tensor: torch.Tensor - shape (3, H, W), dtype (torch.float32)
        """
        self.model.eval()
        tensor = tensor.to(self.device)
        tensor = tensor.expand(1, -1, -1, -1)

        with torch.no_grad():
            pred = self.model(tensor)
            normalized = torch.sigmoid(pred)
            binary = (normalized > self.threshold).float()

        return binary


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    a = torch.randint(2, (3, 480, 640), dtype=torch.float32)
    a /= 255.0
    predictor = Predict(args.config)
    prediction = predictor(a)
    print(prediction.shape, prediction.dtype, prediction.min(), prediction.max(), prediction.unique())


