import yaml
import os
import torch
import torchmetrics as tm
import argparse
import json
from src.models.unet_model import UnetModel
from src.models.dataset import CranesDataset
from src.models.utils import load_checkpoint
from torch.utils.data import DataLoader

class EvaluationSession(object):
    def __init__(self, config_path: str, mode='train') -> None:
        super(EvaluationSession, self).__init__()

        with open(config_path) as config_file:
            self.config = yaml.safe_load(config_file)

        self.plots_path = self.config['evaluate_model']['plots_path']
        self.metrics_path = self.config['evaluate_model']['metrics_path']
        """
            todo: threshold should be iterated to find best fitting
        """
        self.threshold = self.config['evaluate_model']['threshold']
        self.preds = preds
        self.target = target

        self.checkpoint = self.config['train_model']['checkpoint_path']
        self.in_channels = self.config['train_model']['in_features']
        self.out_channels = self.config['train_model']['out_features']
        self.dest_height = self.config['train_model']['dest_height']
        self.dest_width = self.config['train_model']['dest_width']

        self.device = self.config['train_model']['device']
        self.model = UnetModel(self.in_channels, self.out_channels).to(self.device)
        load_checkpoint(torch.load(self.checkpoint), self.model)

        if mode == 'train': 
            self.frames_path = self.config['split_dataset']['train_frames_path']
            self.masks_path = self.config['split_dataset']['train_masks_path']
        elif mode == 'test':
            self.frames_path = self.config['split_dataset']['test_frames_path']
            self.masks_path = self.config['split_dataset']['test_masks_path']
        else:
            self.frames_path = self.config['split_dataset']['valid_frames_path']
            self.masks_path = self.config['split_dataset']['valid_masks_path']

        self.preds, self.target = self.__make_preds()
        self.accuracy = self.__accuracy()
        self.precision = self.__precision()
        self.recall = self.__recall()
        self.f1 = self.__f1()
        self.jaccard = self.__jaccard()
        self.__roc_curve()
        #self.__precision_recall_curve()
        self.__confusion_matrix()
    
    def __make_preds(self):
        self.model.eval()
        ds = CranesDataset(self.frames_path, self.masks_path, self.dest_height, self.dest_width)
        loader = DataLoader(ds, 10000, False)
        for data, target in loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)
                normalized = torch.sigmoid(pred)
                binary = (normalized > self.threshold).float()
                squeezed = torch.squeeze(binary)

        return squeezed.type(torch.uint8), target.type(torch.uint8)

    def __accuracy(self):
        """
            correct pixels over total number of pixels
        """
        accuracy_fn = tm.Accuracy(num_classes=2, average='weighted', mdmc_average='global')
        accuracy = accuracy_fn(self.preds, self.target)
        with open(os.path.join(self.metrics_path, 'accuracy.json'), 'w') as f:
            json.dump({'accuracy': float(accuracy)}, f)
    
    def __precision(self):
        """
            correct pixels over number of guessed as correct
        """
        precision_fn = tm.Precision(num_classes=2, average='weighted', mdmc_average='global')
        precision = precision_fn(self.preds, self.target)
        with open(os.path.join(self.metrics_path, 'precision.json'), 'w') as f:
                    json.dump({'precision': float(precision)}, f)
    
    def __recall(self):
        """
            correct pixels over number of actual correct pixels
        """
        recall_fn = tm.Recall(num_classes=2, average='weighted', mdmc_average='global')
        recall = recall_fn(self.preds, self.target)
        with open(os.path.join(self.metrics_path, 'recall.json'), 'w') as f:
                    json.dump({'recall': float(recall)}, f)
    
    def __f1(self):
        """
            f1 score or dice coefficient -
                harmonic mean over recall and precision
        """
        f1_fn = tm.F1Score(num_classes=2, average='weighted', mdmc_average='global')
        f1 = f1_fn(self.preds, self.target)
        with open(os.path.join(self.metrics_path, 'f1.json'), 'w') as f:
                    json.dump({'f1': float(f1)}, f)
    
    def __jaccard(self):
        """
            jaccard index or intersection over union (IOU)
        """
        jaccard_fn = tm.JaccardIndex(num_classes=2, average='weighted')
        jaccard = jaccard_fn(self.preds, self.target)
        with open(os.path.join(self.metrics_path, 'jaccard_index.json'), 'w') as f:
                    json.dump({'jaccard_index': float(jaccard)}, f)

    
    def __roc_curve(self):
        """
            recall to specifity curve
        """
        roc = tm.ROC()
        timeline = []
        for fpr, tpr, thresh in roc(self.preds, self.target):
            timeline.append({'fpr': float(fpr), 'tpr': float(tpr), 'threshold': float(thresh)})
        with open(os.path.join(self.plots_path, 'roc.json'), 'w') as f:
                    json.dump({'roc': timeline}, f)
        
    def __precision_recall_curve(self):
        #todo: precision recall curve
        """  
            precision to recall curve
        """
        prc = tm.PrecisionRecallCurve()
        timeline = []
        #print(prc(self.preds, self.target))
        #for precison, recall, thresh in prc(self.preds, self.target):
        #    timeline.append({'precision': float(precison), 'recall': float(recall), 'threshold': float(thresh)})
        #with open(os.path.join(self.plots_path, 'prc.json'), 'w') as f:
        #            json.dump({'prc': timeline}, f)

    def __confusion_matrix(self):
        cm_fn = tm.ConfusionMatrix(num_classes=2)
        cm = cm_fn(self.preds, self.target)
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]

        timeline = []
        [timeline.append({'actual': 1, 'predicted': 1}) for _ in range(tp)]
        [timeline.append({'actual': 1, 'predicted': 0}) for _ in range(fn)]
        [timeline.append({'actual': 0, 'predicted': 1}) for _ in range(fp)]
        [timeline.append({'actual': 0, 'predicted': 0}) for _ in range(tn)]

        with open(os.path.join(self.plots_path, 'confusion_matrix.json'), 'w') as f:
            json.dump(timeline, f)
        

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--mode', dest='mode')
    args = args_parser.parse_args()

    preds = torch.randint(2, (1,100,100))
    target = torch.randint(2, (1,100,100))

    es = EvaluationSession(args.config, args.mode)
    """
    print('accuracy: '+str(es.accuracy))
    print('precision: '+ str(es.precision))
    print('recall: '+ str(es.recall))
    print('f1: '+ str(es.f1))
    print('jaccard: '+ str(es.jaccard))
    """


