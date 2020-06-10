import os
import json
import torch, torchvision.transforms as transforms
import numpy

from sklearn.metrics import roc_auc_score

from Data_Class.Data_Set import DataSet
from Data_Class.Class_DS_Net import DS_Net


class Path:
    # Public Variables..
    # Divide Constant and String Variable
    n_classes = 14
    BATCH_SIZE = 64

    # Make Variable that can available to put strings.
    def __init__(self, n_classes):
        self.N_CLASSES = n_classes
        self.CKPT_PATH = ''
        self.CLASS_NAMES = ['']
        self.DATA_DIR = ''
        self.TEST_IMAGE_LIST = ''

    # Realizing how to use Tensor is really good I guess..
    def Calculation(self):
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack([
                    transforms.ToTensor()(crop) for crop in crops
                ])
            ),
            transforms.Lambda(
                lambda crops: torch.stack([
                    normalize(crop) for crop in crops
                ])
            )
        ])


if __name__ == "__main__":
    path = Path(transforms)

    def process(image_list=None, auroc=False):
        if image_list is None:
            image_list = path.TEST_IMAGE_LIST

        # Make sure that you should not use PATH
        # Because PATH only can import Public Variables.
        test_dataset = DataSet(
            data_dir=path.DATA_DIR,
            image_list=image_list,
            transform=transforms
        )

        torch.backends.cudnn.benchmark = True
