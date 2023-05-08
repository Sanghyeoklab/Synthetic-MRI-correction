import torch
import torch.nn as nn 
import torch.optim as optim
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Dataset import Loader
import Transform
from Model.Create_model import Create_model

class get_lossfunction:
    def __init__(self, args):
        self.args = args
    def __call__(self, x, y, mask = None):
        if(self.args.lossfunction == "MSE"):
            if mask is not None:
                return torch.mean((x[mask > 128] - y[mask > 128]) ** 2)
            else:
                return torch.mean((x - y) ** 2)
        elif(self.args.lossfunction == "MAE"):
            if mask is not None:
                return torch.mean(torch.abs(x[mask > 128] - y[mask > 128]))
            else:
                return torch.mean(torch.abs(x - y))

def Test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.device)
    test_transform = Transform.Compose([
        Transform.ToTensor(),
        Transform.Resize(args),
        Transform.Normalize(args)
    ])
    
    args.BatchSize = 1
    TestLoader = Loader(args, Mode = "Test", transform = test_transform)
    model = Create_model(args)
    model.load_state_dict(torch.load(args.Save + "Network_Parameter/Best.pth"))
    model = torch.nn.DataParallel(model.cuda())
    
    
    criteria = get_lossfunction(args)
    
    total_loss = 0
    fid = open(args.Save + "test.log", "w")
    fid.write("Loss per iteration : \n")
    with torch.no_grad():
        for i, data in enumerate(tqdm(TestLoader)):
            x = data["input imgs"].cuda(non_blocking = True)
            y = data["target imgs"].cuda(non_blocking = True)
            output = model(x)
            if "masks" in data.keys():
                loss = criteria(output, y, data['masks'])
            else:
                loss = criteria(output, y)
            fid.write("Itertation number : " + str(i + 1) + " ==> " + str(loss.item()) + "\n")
            total_loss += loss.item()
    fid.write("\n\nLoss average : " + str(total_loss / len(TestLoader)) + "\n")
    fid.close()