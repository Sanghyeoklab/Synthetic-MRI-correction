import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Dataset import Loader
import Transform
from Model.Create_model import Create_model

def get_optimizer(model, args):
    if args.Optimizer == "SGD":
        return optim.SGD(model.parameters(), lr = args.learningRate, weight_decay = args.weight_decay)
    elif args.Optimizer == "Adam":
        return optim.Adam(model.parameters(), lr = args.learningRate, weight_decay = args.weight_decay)

    
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
    
    
def get_scheduler(args, optimizer):
    if args.Scheduler == "ReduceLRONPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 10, eps = args.lr_limit)
    
def Train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.device)
    train_transform = Transform.Compose([
        Transform.ToTensor(),
        Transform.RandRotate(args),
        Transform.Resize(args),
        Transform.Normalize(args)
    ])
    
    validation_transform = Transform.Compose([
        Transform.ToTensor(),
        Transform.Resize(args),
        Transform.Normalize(args)
    ])
    
    TrainLoader = Loader(args, Mode = "Train", transform = train_transform)
    ValidationLoader = Loader(args, Mode = "Validation", transform = validation_transform)
    model = Create_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)
    criteria = get_lossfunction(args)
    model = torch.nn.DataParallel(model.cuda())
    
    if(args.Save[:-1] != "/"):
        args.Save = args.Save + "/"
    os.makedirs(args.Save + "Network_Parameter/", exist_ok = True)
    
    if os.path.exists(args.Save + "Model"):
        shutil.rmtree(args.Save + "Model")
    shutil.copytree("Model", args.Save + "Model")
    
    writer = SummaryWriter(args.Save)
    digit = int(np.log10(args.Epoch))
    
    
    best = np.inf
    
    for epoch in tqdm(range(1, args.Epoch + 1)):
        train_loss = 0
        model.module.train()
        model.module.cuda()
        for data in TrainLoader:
            x = data["input imgs"].cuda(non_blocking = True)
            y = data["target imgs"].cuda(non_blocking = True)
            optimizer.zero_grad()
            output = model(x)
            if "masks" in data.keys():
                loss = criteria(output, y, data['masks'])
            else:
                loss = criteria(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss / len(TrainLoader)
        validation_loss = 0
        model.module.eval()
        with torch.no_grad():
            for data in ValidationLoader:
                x = data["input imgs"].cuda(non_blocking = True)
                y = data["target imgs"].cuda(non_blocking = True)
                output = model(x)
                if "masks" in data.keys():
                    loss = criteria(output, y, data['masks'])
                else:
                    loss = criteria(output, y)
                validation_loss += loss / len(ValidationLoader)
        if(validation_loss < best):
            best = validation_loss
            torch.save(model.module.cpu().state_dict(), args.Save + "Network_Parameter/Best.pth")
        if(args.Scheduler == "ReduceLRONPlateau"):
            scheduler.step(train_loss)
        elif(scheduler is not None):
            scheduler.step()
        torch.save(model.module.cpu().state_dict(), args.Save + "Network_Parameter/" + "0" * (digit - int(np.log10(epoch))) + str(epoch) + ".pth")
        writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", validation_loss, epoch)
