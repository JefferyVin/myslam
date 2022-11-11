#!/usr/bin/env python3

import cv2
import numpy as np
from extractor import Extractor

W = 1920 // 2
H = 1080 // 2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
orb = cv2.ORB_create()
#print(dir(orb))



        
        

fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    kps, des = fe.extract(img)
    #kps, des = orb.detectAndCompute(img,None)
    for p in kps:
        u, v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (u,v), color = (0, 255, 0), radius = 3)

    # for p in kps:
    #     u, v = map(lambda x: int(round(x)), p[0])
    #     cv2.circle(img, (u,v), color = (0, 255, 0), radius = 3)

    cv2.imshow('image', img)
    #print(img.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture('twitchslam/videos/test_countryroad.mp4')
    #print(cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)

            cv2.waitKey(1)
        else:
            break

    cap.release()
 
    # Closes all the frames
    cv2.destroyWindow('image')
    cv2.waitKey(1)