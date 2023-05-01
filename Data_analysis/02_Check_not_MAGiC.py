import numpy as np
import os
from tqdm import tqdm
import Analysis_Util
import argparse
import pydicom
from collections import defaultdict


parser = argparse.ArgumentParser(prog = 'MAGiC', description = 'Check defective dicom')
parser.add_argument('--load_folder', help = 'load every dicom folders', default = 'Data/')
args = parser.parse_args()
load_folder = args.load_folder
if(load_folder[-1] != "/"):
    load_folder = load_folder + "/"

lists = [load_folder + l + "/" for l in os.listdir(load_folder)]
dic = defaultdict(int)
find = ["MAGiC STIR", "MAGiC T2", "MAGiC PD", "MAGiC T1FLAIR", "MAGiC T1", "MAGiC IR", "MAGiC T2FLAIR"]
not_MAGiC = []

for l in tqdm(lists):
    index = Analysis_Util.get_list(l)
    condition = 0
    local_dic = defaultdict(int)
    for i in index:
        files = os.listdir(i)
        dcm = pydicom.read_file(i + files[0])
        if(dcm.SeriesDescription in find):
            local_dic[dcm.SeriesDescription] = 1
    if(np.sum(np.array(list(local_dic.values()))) == 6):
        dic["Total"] = dic["Total"] + 1
    else:
        not_MAGiC.append(l)
    
for k, v in dic.items():
    print(k, " : ", v)

print("Data reject : ")
for n in not_MAGiC:
    print(n)