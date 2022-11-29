import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class Extractor(object):
    GX = 16//2
    GY = 12//2
    
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, img):

        #detection
        corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), 3000, 0.01, 3)
        
        #extraction
        kps = [cv2.KeyPoint(crd[0][0], crd[0][1], 13) for crd in corners]
        kps, des = self.orb.compute(img, kps)
        
        #matching
        ret = []
        
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k = 2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
        
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)

        #filter
            model, inliers = ransac((ret[:, 0], ret[:, 1]), FundamentalMatrixTransform,
                                     min_samples = 8, residual_threshold = 1, max_trials = 100)
            #print(sum(inliers))
            ret = ret[inliers]

        self.last = {'kps': kps, 'des': des}
        #print(type(corners))
        return ret