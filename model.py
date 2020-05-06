import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet


'''
class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, 128)
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
'''

class Net:
    input_sizes = {
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
        'efficientnet-b3': 300,
        'efficientnet-b4': 380,
        'efficientnet-b5': 456,
        'efficientnet-b6': 528,
        'efficientnet-b7': 600,
    }
    
    def __init__(self, name, num_classes, feature_extract, use_pretrained=True, advprop=True):
        self.name = name
        if use_pretrained:
            self.model_ft = EfficientNet.from_pretrained(name, advprop=True)
        else:
            self.model_ft = EfficientNet.from_name(name)

        if feature_extract:
            for param in self.model_ft.parameters():
                param.requires_grad = False

        self.num_ftrs = self.model_ft._fc.in_features
        self.input_size = Net.input_sizes[name]
        
    
        
