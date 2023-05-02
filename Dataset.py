import cv2
import numpy as np
import Util
from torch.utils.data import DataLoader

def dic2info(dictionary, keys):
    info = []
    for index in range(len(dictionary[keys[0]])):
        string = ""
        for k in keys: 
            if(string == ""):
                string = dictionary[k][index] 
            else:
                string = string + " " + dictionary[k][index] 
        info.append(string)
    return info

def list2dcm(list):
    files = list.split(" ")
    imgs = np.array([Util.read_dicom(f) for f in files]) 
    return imgs


class Synthetic_dataset(object):
    def __init__(self, args, Mode = "Train", transform = None):
        assert Mode in ["Train", "Validation", "Test"], "Mode must be Train, Validation or Test"
        dictionary = Util.yaml2dic(args.Dataset)[Mode]
        self.input_list = dic2info(dictionary, args.Input)
        self.output_list = dic2info(dictionary, args.Output)
        if args.Mask is not None and args.Mask == True:
            self.mask_list = dic2info(dictionary, ["Mask"])
        else:
            self.mask_list = None
        self.transform = transform
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        input_imgs = list2dcm(self.input_list[idx])
        output_imgs = list2dcm(self.output_list[idx])
        if self.mask_list is not None: 
            mask = np.expand_dims(cv2.imread(self.mask_list[idx], -1), 0)
        else:
            mask = None
        if self.transform is not None: 
            input_imgs, output_imgs, mask = self.transform(input_imgs, output_imgs, mask)
        
        return input_imgs, output_imgs, mask
    
def Loader(args, Mode = "Train", transform = None):
    dataset = Synthetic_dataset(args, Mode, transform)
    shuffle = True if Mode == "Train" else False
    return DataLoader(dataset, batch_size = args.BatchSize, num_workers = args.num_workers, drop_last=True, shuffle = shuffle)