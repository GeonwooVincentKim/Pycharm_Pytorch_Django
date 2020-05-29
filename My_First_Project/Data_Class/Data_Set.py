import os
import torch
from torch.utils.data import Dataset as torch_data
import numpy as np

from PIL import Image


class DataSet(torch_data):
    def __init__(self, data_dir, image_list, transform=None):

        """
       :param data_dir: Image Directory are access to the Path.
       :param image_list: Containing images with corresponding labels,
                          then, access to Path and Path Files.
       :param transform: Sample are applied from optional transform.
       """

        image_names = []
        labels = []

        if isinstance(image_list, (list, tuple)):
            image_names = image_list
            labels = np.zeros([1, 14])
        else:
            with open(image_list, "r") as f:
                for line in f:
                    items = line.split()
                    image_name = items[0]
                    label = items[1:]
                    label = [int(i) for i in label]

                    image_name = os.path.join(data_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """

        :param index: Index Item
        :return: Image and Index labels
        """

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.FloatTensor(label).cpu()

    def __len__(self):
        return len(self.image_names)
