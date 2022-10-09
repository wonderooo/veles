from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import ResNet101_Weights
from torch.nn import CrossEntropyLoss
from src.models.dataset import CranesDataset
import torch

ds = CranesDataset('../../../../../yolov7-seg/yolov7_d2/datasets/train/images', '../../../../../yolov7-seg/yolov7_d2/datasets/train/images', 480, 640)
i, m = ds.__getitem__(0)

model = deeplabv3_resnet101(num_classes=3, weights_backbone=ResNet101_Weights.IMAGENET1K_V2).train()
loss_fn = CrossEntropyLoss()
print(i.shape, m.shape)

batch_i = torch.randint(2, (2, 3, 480, 640), dtype=torch.float32)
batch_m = torch.randint(4, (2, 480, 640), dtype=torch.long)
print(batch_m.max())
pred = model(batch_i)
loss = loss_fn(pred['out'], batch_m)
print(loss)
