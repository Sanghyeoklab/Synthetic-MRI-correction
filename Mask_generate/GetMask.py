import cv2
import argparse
import pydicom
import Analysis_Util

parser = argparse.ArgumentParser(prog = 'Synthetic MRI artifact correction', description = 'Make mask')
parser.add_argument('--load_dicom', help = 'load dicom', default = 'Data/dicom.dcm')
parser.add_argument('--save_file', help = 'save mask', default = 'Save/mask.jpg')
args = parser.parse_args()

dcm = pydicom.read_file(args.load_dicom)
img = Analysis_Util.img_load(dcm)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
make_roi = Analysis_Util.click_event(img)
cv2.setMouseCallback('image', make_roi.click_event)
cv2.waitKey(0)
if make_roi.mask is not None:
    cv2.imwrite(args.save_file, make_roi.mask)
cv2.destroyAllWindows()