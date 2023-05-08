import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class Compose(object):
    def __init__(self, transform):
        self.transform = transform 
    def __call__(self, input, label, mask):
        for t in self.transform: 
            input, label, mask = t(input, label, mask)
        return input, label, mask
    
class ToTensor(object):
    def __call__(self, input, label, mask):
        input = torch.Tensor(input)
        label = torch.Tensor(label) 
        mask = torch.Tensor(mask)
        return input, label, mask

class Normalize(object):
    def __init__(self, args):
        self.inputNormalize = []
        self.labelNormalize = []
        for k in args.Input:
            self.inputNormalize.append(args.Normalize[k])
        for k in args.Output:
            self.labelNormalize.append(args.Normalize[k])
    def __call__(self, input, label, mask):
        for i, [mean, std] in enumerate(self.inputNormalize): 
            input[i] = (input[i] - mean) / std
        for i, [mean, std] in enumerate(self.labelNormalize): 
            label[i] = (label[i] - mean) / std
        return input, label, mask

class Resize(object):
    def __init__(self, args):
        self.imgSize = args.imgSize
    def __call__(self, input, label, mask):
        input = F.resize(input, self.imgSize, interpolation = InterpolationMode.BICUBIC, antialias = True)
        label = F.resize(label, self.imgSize, interpolation = InterpolationMode.BICUBIC, antialias = True)
        mask = F.resize(mask, self.imgSize, interpolation = InterpolationMode.NEAREST, antialias = True)
        return input, label, mask
    
    
class RandRotate(object):
    def __init__(self, args):
        self.rot_param = args.Rotate
    def __call__(self, input, label, mask):
        random = torch.rand(1).item() * (self.rot_param[1] - self.rot_param[0]) + self.rot_param[0]
        input = F.rotate(input, random, interpolation = InterpolationMode.BILINEAR) 
        label = F.rotate(label, random, interpolation = InterpolationMode.BILINEAR) 
        mask = F.rotate(label, random, interpolation = InterpolationMode.NEAREST) 
        return input, label, mask