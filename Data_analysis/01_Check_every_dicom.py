import os
from tqdm import tqdm
import Analysis_Util
import argparse
import pydicom
from collections import defaultdict

parser = argparse.ArgumentParser(prog = 'MAGiC', description = 'Check dicom series')
parser.add_argument('--load_folder', help = 'load every dicom folders', default = 'Data/')
args = parser.parse_args()

load_folder = args.load_folder
if(load_folder[-1] != "/"):
    load_folder = load_folder + "/"

lists = [load_folder + l + "/" for l in os.listdir(load_folder)]
dic = defaultdict(int)
key_length = 0
value_length = 0
for l in tqdm(lists):
    index = Analysis_Util.get_list(l)
    for i in index:
        files = os.listdir(i)
        dcm = pydicom.read_file(i + files[0])
        dic[dcm.SeriesDescription] = dic[dcm.SeriesDescription] + 1
        key_length = max(key_length, len(dcm.SeriesDescription))
        value_length = max(value_length, dic[dcm.SeriesDescription])
for k, v in dic.items():
    print("%-50s\t : \t%10s"%(k, str(v)))