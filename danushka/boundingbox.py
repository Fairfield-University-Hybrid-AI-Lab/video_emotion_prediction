import numpy as np
from cv2 import cv2
import os.path



VIDEO_NUMBER = 102
BOUNDING_BOX_PATH = ".\\bounding-boxes"
ANNOTATIONS_PATH =  ".\\annotations"

def main():
    #load annotations
    valence_annotations = load_annotations(ANNOTATIONS_PATH +"\\valence\\"+ str(VIDEO_NUMBER)+".txt")
    arousal_annotations = load_annotations(ANNOTATIONS_PATH +"\\arousal\\"+ str(VIDEO_NUMBER)+".txt")

    frame_number = 0
    cap = cv2.VideoCapture(".\\videos\\"+str(VIDEO_NUMBER)+".avi")
    while(cap.isOpened()):
        frame_number += 1

        #load bounding box coords
        bounding_box = load_points(BOUNDING_BOX_PATH +"\\"+ str(VIDEO_NUMBER)+"\\"+str(frame_number)+".pts")
        if not bounding_box:
            continue
        frame = cap.read()[1]       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #write bounding box on image
        pts = np.array([[bounding_box[0][0],bounding_box[0][1]],[bounding_box[1][0],bounding_box[1][1]],[bounding_box[2][0],bounding_box[2][1]],[bounding_box[3][0],bounding_box[3][1]]], np.int32)
        pts = pts.reshape((-1,1,2))
        img = cv2.polylines(gray,[pts],True,(0,255,255))

        # Write arousal, valence on image
        cv2.putText(img,"valence: "+ valence_annotations[frame_number], (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(img,"arousal: "+ arousal_annotations[frame_number], (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        cv2.imshow('frame',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


def load_points(path):
    """
    load the coordinates from a point cloud file
    INPUT: path of file
    OUTPUT: coordinates as a two dimensional list
    """
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = [rows.strip() for rows in f]
        
        #find start and end of points
        head = rows.index('{') + 1
        tail = rows.index('}')

        raw_points = rows[head:tail]
        coords_set = [point.split() for point in raw_points]

        points = [tuple([float(point) for point in coords]) for coords in coords_set]
        return points

def load_annotations(path):
    """
    load the annotations from a text file line by line
    INPUT: path of file
    OUTPUT: list of annotations
    """
    if not os.path.isfile(path):
        return None
    with open(path) as file:
        lines = [line.strip() for line in file]
    return lines

if __name__ == '__main__':
    main()