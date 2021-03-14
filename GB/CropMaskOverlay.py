import numpy as np
from cv2 import cv2
import os.path

        
 
def CROPMASK(img,pts,lndmark,clr) : 
    
    mydict = {
        'bottomhalf': pts[0:16],
        'eyebrows': pts[17:26],
        'nose': pts[27:35],
        'eyes': pts[36:47],
        'lips': pts[48:60],
        'topandnose': pts[17:35],
        'tophalf': pts[17:47],
        'tophalfconc': np.concatenate((pts[27:35],pts[17:26],pts[36:47])),
        'all' : pts
    }
    
    
    maskpts = mydict[lndmark]
    
    cv2.drawContours(img, [maskpts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    maskedimg = cv2.fillPoly(img ,mydict[lndmark], (0,0,clr) )

    
    # rect = cv2.boundingRect(maskpts)
    # x,y,w,h = rect
    # rectMask = img[y:y+h, x:x+w].copy()
    # mask = np.zeros(rectMask.shape[:2], np.uint8)
    # cv2.drawContours(img, mask, -1, (255, 255, 255), -1, cv2.LINE_AA)
    

    return maskedimg

    