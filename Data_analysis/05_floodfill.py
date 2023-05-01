import cv2
import Analysis_Util
import argparse

parser = argparse.ArgumentParser(prog = 'MAGiC', description = 'Reinforce mask')
parser.add_argument('--load_mask', help = 'load mask', default = 'Data/mask.jpg')
parser.add_argument('--save_mask', help = 'save mask', default = 'Data/mask.jpg')
args = parser.parse_args()

mask = cv2.imread(args.load_mask, -1)
mask = (mask > 128) * 255
mask = Analysis_Util.flood_fill(mask, [0, 0])
cv2.imwrite(args.save_mask, mask)