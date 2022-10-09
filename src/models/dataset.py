import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CranesDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, \
        dest_height: int, dest_width: int) -> None:

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = list(filter(lambda x: x.endswith('.png') and not x.endswith('m.png'), \
            os.listdir(image_dir)))
        self.transform = A.Compose(
            [
                #A.Rotate(limit=8, p=0.3),
                #A.CropNonEmptyMaskIfExists(int(dest_height*(3/2)), int(dest_width*(3/2)), [0]),
                #A.RandomBrightnessContrast(p=0.2),
                A.Resize(dest_height, dest_width),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask != 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = CranesDataset('../../../../yolov7-seg/yolov7_d2/datasets/train/images', '../../../../yolov7-seg/yolov7_d2/datasets/train/outlined', 480, 640)
    i, m = ds.__getitem__(0)
    plt.imshow(m)
    plt.show()
    print(i.shape, m.shape, i.dtype, m.dtype, i.max(), m.max(), m.unique())
    print(len(ds))
