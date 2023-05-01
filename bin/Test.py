import torch
import torch.nn as nn 
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Dataset import Loader
import Transform
from Model.Create_model import Create_model

def get_lossfunction(args):
    if args.lossfunction == "MSE":
        return nn.MSELoss()
    
def Test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.device)
    test_transform = Transform.Compose([
        Transform.ToTensor(),
        Transform.Resize(args),
        Transform.Normalize(args)
    ])
    
    TestLoader = Loader(args, Mode = "Test", transform = test_transform)
    model = Create_model(args)
    model = torch.nn.DataParallel(model.cuda())
    model.load_state_dict(args.Save + "Network_Parameter/Best.pth")
    
    criteria = get_lossfunction(args)
    
    with torch.no_grad():
        for x, y, mask in TestLoader:
            x = x.cuda(non_blocking = True)
            y = y.cuda(non_blocking = True)
            mask = mask.cuda(non_blocking = True)
            output = model(x)
            
            if mask is not None:
                output *= (mask > 128)
                y  *= (mask > 128)
            loss = criteria(output, y)
