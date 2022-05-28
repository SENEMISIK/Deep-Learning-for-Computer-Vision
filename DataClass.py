import torch
import torchvision
import os

class TSintel(torchvision.datasets.Sintel):
    def __init__(self, root):
        super().__init__(root=root)
        
    def __getitem__(self, index):
        img1, img2, flow = super().__getitem__(index)
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        return img1, img2, flow
    
class TKitti(torchvision.datasets.KittiFlow):
    def __init__(self, root):
        super().__init__(root=root)
        
    def __getitem__(self, index):
        img1, img2, flow, valid_flow_mask = super().__getitem__(index)
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        return img1, img2, flow, valid_flow_mask
    
class TFlyingChairs(torchvision.datasets.FlyingChairs):
    def __init__(self, root):
        super().__init__(root=root)
        
    def __getitem__(self, index):
        img1, img2, flow, valid_flow_mask = super().__getitem__(index)
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        return img1, img2, flow, valid_flow_mask

