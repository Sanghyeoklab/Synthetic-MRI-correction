import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class Compose(object):
    def __init__(self, transform):
        self.transform = transform 
        
    def __call__(self, input, output, mask):
        for t in self.transform: 
            input, output, mask = t(input, output, mask)
        return input, output, mask 
    
class ToTensor(object):
    def __call__(self, input, output, mask):
        input = torch.Tensor(input)
        output = torch.Tensor(output) 
        mask = torch.Tensor(mask) 
        return input, output, mask

class Normalize(object):
    def __init__(self, args):
        self.inputNormalize = []
        self.outputNormalize = []
        for k in args.Input:
            self.inputNormalize.append(args.Normalize[k])
        for k in args.Output:
            self.outputNormalize.append(args.Normalize[k])
    def __call__(self, input, output, mask):
        for i in range(input.shape[0]): 
            input[i] = (input[i] - self.inputNormalize[i][0]) / self.inputNormalize[i][1]
        for i in range(output.shape[0]): 
            output[i] = (output[i] - self.outputNormalize[i][0]) / self.outputNormalize[i][1]
        return input, output, mask

class Resize(object):
    def __init__(self, args):
        self.imgSize = args.imgSize
    def __call__(self, input, output, mask):
        input = F.resize(input, self.imgSize, interpolation = InterpolationMode.BICUBIC)
        output = F.resize(output, self.imgSize, interpolation = InterpolationMode.BICUBIC)
        mask = F.resize(mask, self.imgSize, interpolation = InterpolationMode.NEAREST)
        return input, output, mask
    
    
class RandRotate(object):
    def __init__(self, args):
        self.rot_param = args.Rotate
    def __call__(self, input, output, mask):
        random = torch.rand(1).item() * (self.rot_param[1] - self.rot_param[0]) + self.rot_param[0]
        input = F.rotate(input, random, interpolation = InterpolationMode.BILINEAR) 
        output = F.rotate(output, random, interpolation = InterpolationMode.BILINEAR) 
        mask = F.rotate(mask, random, interpolation = InterpolationMode.BILINEAR) 
        return input, output, mask