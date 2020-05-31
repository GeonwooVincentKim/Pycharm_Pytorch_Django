import torch
import torchvision


class DS_Net(torch.nn.Module):
    """
        Model modified.

        The architecture of our model is the same as standard DS_Net
        except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DS_Net, self).__init__()
        self.dsnet = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.dsnet.classifier.in_features
        self.dsnet.classfier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, out_size),  # Draw as Linear Graph.
            torch.nn.Sigmoid()  # Sigmoid Function
        )

    def forward(self, x):
        x = self.dsnet(x)
        return x
