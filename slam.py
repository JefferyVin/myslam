#!/usr/bin/env python3

import cv2

W = 1920 // 2
H = 1080 // 2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

def process_frame(img):
    img = cv2.resize(img, (W, H))
    cv2.imshow('image', img)
    print(img.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/test_countryroad.mp4')
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