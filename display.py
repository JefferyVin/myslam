import cv2

class Display(object):
    def __init__(self, W, H):
        self.W, self.H = W, H
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Resized_Window", self.W, self.H)
    
    def show(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()