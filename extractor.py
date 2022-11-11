import cv2
import numpy as np

class Extractor(object):
    GX = 16//2
    GY = 12//2
    
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, img):
        # Attempt at SIFT (problem too slow)
        # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        # return sift.detectAndCompute(img, None)
        # 
        # Attempt to use grid (problem the clusters are too much)        
        # sy = img.shape[0]//self.GY
        # sx = img.shape[1]//self.GX
        # akp = []
        # for ry in range(0, img.shape[0], sy):
        #     for rx in range(0, img.shape[1], sx):
        #         img_chunk = img[ry:ry+sy, rx:rx+sx]
        #         chunkKeypoints = self.orb.detect(img_chunk, None)
        #         for p in chunkKeypoints:
        #             p.pt = (p.pt[0] + rx, p.pt[1]+ry)
        #             akp.append(p)
        # return akp

        #detection
        corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), 3000, 0.01, 3)
        
        #extraction
        kps = [cv2.KeyPoint(crd[0][0], crd[0][1], 13) for crd in corners]
        kps, des = self.orb.compute(img, kps)
        
        #matching
        matches = None
        
        if self.last is not None:
            matches = self.bf.match(des, self.last['des'])
            #print(matches)
        self.last = {'kps': kps, 'des': des}
        #print(type(corners))
        return kps, des, matches