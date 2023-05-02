import argparse
import Util
from Model.Create_model import Create_model
import os
import torch
import numpy as np
from tqdm import tqdm
class Compose(object):
    def __init__(self, transform):
        self.transform = transform 
        
    def __call__(self, input):
        for t in self.transform: 
            input = t(input)
        return input
    
class ToTensor(object):
    def __call__(self, input):
        input = torch.Tensor(input).unsqueeze(0)
        return input

class Normalize(object):
    def __init__(self, args):
        self.inputNormalize = []
        for k in args.Input:
            self.inputNormalize.append(args.Normalize[k])
    def __call__(self, input):
        for i in range(input.shape[0]): 
            input[i] = (input[i] - self.inputNormalize[i][0]) / self.inputNormalize[i][1]
        return input



parser = argparse.ArgumentParser(prog = 'Synthetic MRI artifact correction', description = 'inference')
parser.add_argument('--config', help = 'Get parameters', default = 'Config/config.yaml')
parser.add_argument('--load_folder', help = 'Get dicom bundle folder path', default = '../Data/')
parser.add_argument('--Save_folder', help = 'Dicom save folder path', default = 'inference/')
args = parser.parse_args()

load_folder = args.load_folder
Save_folder = args.Save_folder
if load_folder[-1] != "/":
    load_folder += "/"
if Save_folder[-1] != "/":
    Save_folder += "/"
os.makedirs(Save_folder, exist_ok=True)
args = Util.make_args(Util.yaml2dic(args.config))

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.device)


model = Create_model(args)
model = torch.nn.DataParallel(model.cuda())
print(args.Save + "Network_Parameter/Best.pth")
model.load_state_dict(torch.load(args.Save + "Network_Parameter/Best.pth"))
lists = os.listdir(load_folder + args.Input_path["T2 FLAIR"])

Transform = Compose([
    ToTensor(),
    Normalize(args)
])

for l in tqdm(lists):
    imgs = []
    for method in args.Input:
        imgs.append(Util.read_dicom(load_folder + args.Input_path[method] + "/" + l))
    imgs = Transform(np.array(imgs)).cuda()
    with torch.no_grad():
        output = model(imgs).cpu().numpy()[0, 0]
    output = output * args.Normalize["Convention method"][1] + args.Normalize["Convention method"][0]
    Util.save_dicom(output, Save_folder + l, load_folder + args.Input_path["Convention method"] + l)