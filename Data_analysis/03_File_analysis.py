import os
from tqdm import tqdm
import pydicom
from collections import defaultdict
import Analysis_Util
import argparse


parser = argparse.ArgumentParser(prog = 'MAGiC', description = 'Dicom analysis')
parser.add_argument('--load_folder', help = 'load every dicom folders', default = 'Data/"')
parser.add_argument('--save_folder', help = 'Save dicom folder path', default = 'Save/')
args = parser.parse_args()
load_folder = args.load_folder
save_folder = args.save_folder

if(load_folder[-1] != "/"):
    load_folder = load_folder + "/"
if(save_folder[-1] != "/"):
    save_folder = save_folder + "/"

lists = [load_folder + l + "/" for l in os.listdir(load_folder)]
dic = defaultdict(int)
find = ["MAGiC STIR", "MAGiC T2", "MAGiC PD", "MAGiC T1FLAIR", "MAGiC T1", "MAGiC T2FLAIR", "Ax T2 FLAIR", "MAGiC IR"]
not_MAGiC = []
for l in tqdm(lists):
    index = Analysis_Util.get_list(l)
    condition = 0
    local_dic = defaultdict(int)
    os.makedirs(save_folder + l, exist_ok = True)
    for f in find[:-1]:
        os.makedirs(save_folder + l + "/" + f, exist_ok = True)
    for i in index:
        files = os.listdir(i)
        for f in files:
            dcm = pydicom.read_file(i + f)
            if(dcm.SeriesDescription in find):
                if dcm.SeriesDescription == "MAGiC IR":
                    dcm.SeriesDescription = "MAGiC T2FLAIR"
                dcm.save_as(save_folder + l + dcm.SeriesDescription + "/" + f)