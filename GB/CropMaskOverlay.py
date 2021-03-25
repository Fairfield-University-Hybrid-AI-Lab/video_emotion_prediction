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
    cv2.drawContours(img, [maskpts], -1, (clr, clr, clr), -1, cv2.LINE_AA)
    maskedimg = cv2.fillPoly(img ,mydict[lndmark], (clr,clr,clr) )


    return maskedimg


def POINTMASK(img,pts,lndmark,clr) : 
    
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
    maskedimg = cv2.fillPoly(img ,mydict[lndmark], (clr,clr,clr) )


    return maskedimg



def OUTLINERECTMASK(img,pts,lndmark,clr) : 
    
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
    x,y,w,h = cv2.boundingRect(maskpts)
    maskedimg = cv2.rectangle(img, (x, y), (x + w, y + h), (clr,clr,clr), 1)
    

    return maskedimg

def RECTMASK(img,pts,lndmark,clr) : 
    
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
    x,y,w,h = cv2.boundingRect(maskpts)
    maskedimg = cv2.rectangle(img, (x, y), (x + w, y + h), (clr,clr,clr), -1)


    return maskedimg

