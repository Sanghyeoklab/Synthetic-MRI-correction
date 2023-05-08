import yaml
import pydicom
import numpy as np


def NYUL(img1, img2, mask, percen):
    point1 = [np.min(img1[mask > 128])] + list(np.percentile(img1[mask > 128], percen)) + [np.max(img1[mask > 128])]
    point2 = [np.min(img2[mask > 128])] + list(np.percentile(img2[mask > 128], percen)) + [np.max(img2[mask > 128])]
    point1 = [int(p) for p in point1]
    point2 = [int(p) for p in point2]
    y = np.zeros(65536)
    
    slope = (point2[1] - point2[0]) / (point1[1] - point1[0])
    b = -slope * point1[0] + point2[0]
    for j in range(0, point1[0]):
        y[j] = slope * j + b
    
    for i in range(len(point1) - 1):
        slope = (point2[i + 1] - point2[i]) / (point1[i + 1] - point1[i])
        b = -slope * point1[i] + point2[i]
        for j in range(point1[i], point1[i + 1]):
            y[j] = slope * j + b
    
    
    slope = (point2[-1] - point2[-2]) / (point1[-1] - point1[-2])
    b = -slope * point1[-1] + point2[-1]
    for j in range(point1[-1], 65536):
        y[j] = slope * j + b
    
    output = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            output[i, j] = y[int(img1[i, j])]
    
    return output


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
