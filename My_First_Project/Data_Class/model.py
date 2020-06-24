import os

import torch
import torch.utils.data
import torch.backends.cudnn
import torchvision.transforms as transforms

from .Class_DS_Net import DS_Net
from .Data_Set import DataSet


class Path:
    # Public Variables..
    # Divide Constant and String Variable
    n_classes = 14
    BATCH_SIZE = 64

    # Make Variable that can available to put strings.
    def __init__(self, n_classes):
        self.N_CLASSES = n_classes
        self.CKPT_PATH = 'E:/Django/Pytorch_Django/Local_Project/First/My_First_Project/Data_Class/model.pth.tar'
        self.CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                            'Hernia']
        self.DATA_DIR = 'E:/Django/Pytorch_Django/Local_Project/First/My_First_Project/Data_Images/Images'
        self.TEST_IMAGE_LIST = 'E:/Django/Pytorch_Django/Local_Project/First/My_First_Project/Data_Images/labels' \
                               '/short_test_list.txt '

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

        # Load and Initialize the Model
        model = DS_Net(path.n_classes).cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])

        if os.path.isfile(path.CKPT_PATH):
            print("Loading CheckPoint...")
            checkpoint = torch.load(path.CKPT_PATH)
            model.load_state_dict(checkpoint['state_dict'])
            print("Successfully Loaded Checkpoint")
        else:
            print("Could not find checkpoint")

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

        # Initialize the Ground Truth and output Tensor.
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()

        # Switch to evaluate mode
        model.eval()
