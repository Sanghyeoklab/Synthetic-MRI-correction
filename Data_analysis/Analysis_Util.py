import os
import cv2
import numpy as np
from collections import deque

def get_list(folder):
    lists = [folder]
    while(1):
        try:
            files = os.listdir(lists[-1])[0]
            lists.append(lists[-1] + "/" + files)
        except:
            break
    return [lists[-3] + "/" + l + "/" for l in os.listdir(lists[-3])]

def img_load(dcm):
    img = dcm.pixel_array.astype(np.float64)
    min, max = np.percentile(img.reshape(-1), [5, 95])
    img = (img - min) / (max - min) * 255
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


class click_event():
    def __init__(self, img):
        super(click_event, self).__init__()
        self.coordinate = deque()
        self.img = img
        self.mask = None
        if(len(self.img.shape) == 2):
            self.img = np.repeat(np.expand_dims(self.img, 2), 3, 2)
        
    def img_show(self):
        lined = self.img.copy()
        if(len(self.coordinate) == 1):
                lined[self.coordinate[0][1], self.coordinate[0][0]] = [255, 0, 0]
        if(len(self.coordinate) >= 2):
            lined = cv2.polylines(lined, [np.array(self.coordinate)], True, (255, 0, 0), 1)
        cv2.imshow('image', lined)

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                cv2.destroyAllWindows()
                if(len(self.coordinate) >= 3):
                    mask = np.zeros((self.img.shape[0], self.img.shape[1]))
                    mask = cv2.fillConvexPoly(mask, np.array(self.coordinate), 255)
                    self.mask = mask
            else:
                self.coordinate.append([x, y])
                self.img_show()

        if event==cv2.EVENT_RBUTTONDOWN:
            if len(self.coordinate) != 0:
                self.coordinate.pop()
            self.img_show()

def flood_fill(img, coordinate):
    mask = img.copy()
    stack = deque([coordinate])
    direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    while(len(stack)!=0):

        for _ in range(len(stack)):
            row, col = stack.popleft()
            for d in direction:
                if(row + d[0] < 0 or row + d[0] >= img.shape[0] or col + d[1] < 0 or col + d[1] >= img.shape[1]):
                    continue
                if(mask[row + d[0], col + d[1]] != 0):
                    continue
                stack.append([row + d[0], col + d[1]])
                mask[row + d[0], col + d[1]] = 100
    return (mask != 100) * 255