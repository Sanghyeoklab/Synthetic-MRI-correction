import yaml
import pydicom
import numpy as np

def read_dicom(load_path):
    dcm = pydicom.read_file(load_path)
    return np.array(dcm.pixel_array).astype(np.float32)

def MinMax2WWWL(Wmin, Wmax):
    WW = Wmax - Wmin
    WL = (Wmin + Wmax) / 2
    return [WW, WL]

def save_dicom(img, save_path, info_path, Series_description = None, WWWL = None):
    dcm = pydicom.read_file(info_path)
    img[img < 0] = 0
    img[img > 65535] = 65535
    img = np.round(img).astype(np.uint16)
    # dcm.BitsAllocated = 16
    # dcm.BitsStored = 16
    # dcm.HighBit = 15
    # dcm.PixelRepresentation = 1
    dcm.PixelData = img.tobytes()
    if WWWL is not None:
        dcm.WindowCenter = str(WWWL[0])
        dcm.WindowWidth = str(WWWL[1])
        
    if Series_description is not None:
        dcm.SeriesDescription = Series_description
    dcm.save_as(save_path)
    return True

def yaml2dic(path):
    fid = open(path, "r")
    dic = yaml.safe_load(fid)
    fid.close()
    return dic

def make_args(dic):
    class args:
        pass
    for k, v in dic.items():
        setattr(args, k, v)
    return args
