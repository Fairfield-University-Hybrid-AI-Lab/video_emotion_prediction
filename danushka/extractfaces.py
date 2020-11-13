import numpy as np
from cv2 import cv2
import os
import boundingbox 
from pathlib import Path

VIDEOS_PATH = ".\\videos\\"
BOUNDING_BOX_PATH = ".\\bounding-boxes"
ANNOTATIONS_PATH =  ".\\annotations"
VALENCE_CLASS_EDGES = [-1,0,0.1915,1]
AROUSAL_CLASS_EDGES = [-1,0.1315,0.3295,1]

def main():
    video_list = os.listdir(VIDEOS_PATH)
    for video_file in video_list:
        VIDEO_NUMBER = video_file.split('.')[0]
        #load annotations
        valence_annotations = boundingbox.load_annotations(ANNOTATIONS_PATH +"\\valence\\"+ str(VIDEO_NUMBER)+".txt")
        arousal_annotations = boundingbox.load_annotations(ANNOTATIONS_PATH +"\\arousal\\"+ str(VIDEO_NUMBER)+".txt")

        frame_number = -1 #start from zeroth frame
        cap = cv2.VideoCapture(VIDEOS_PATH+str(VIDEO_NUMBER)+".avi")
        while(cap.isOpened()):
            frame_number += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
            #load bounding box coords
            bounding_box = boundingbox.load_points(BOUNDING_BOX_PATH +"\\"+ str(VIDEO_NUMBER)+"\\"+str(frame_number)+".pts")
            if not bounding_box:
                continue
            frame = cap.read()[1]   
            if frame is None: #exit at end of video
                break    
            try:
                valence_value = float(valence_annotations[frame_number])
                arousal_value = float(arousal_annotations[frame_number])
            except:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pts = np.array([[bounding_box[0][0],bounding_box[0][1]],[bounding_box[1][0],bounding_box[1][1]],[bounding_box[2][0],bounding_box[2][1]],[bounding_box[3][0],bounding_box[3][1]]], np.int32)
            pts = pts.reshape((-1,1,2))
            img = cv2.polylines(gray,[pts],True,(0,255,255))
            crop_img = img[ int(bounding_box[0][1]):int(bounding_box[1][1]),int(bounding_box[0][0]):int(bounding_box[2][0]),]
            cv2.imshow("cropped", crop_img)
            
            
            #save crop to path based on valence value
            if valence_value >= VALENCE_CLASS_EDGES[0] and valence_value < VALENCE_CLASS_EDGES[1]:
                Path(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\valence\\low\\").mkdir(parents=True, exist_ok=True)#create directory path if it doesnt exist
                cv2.imwrite(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\valence\\low\\"+str(frame_number)+".png",crop_img)
            elif valence_value >= VALENCE_CLASS_EDGES[1] and valence_value < VALENCE_CLASS_EDGES[2]:
                Path(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\valence\\neutral\\").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\valence\\neutral\\"+str(frame_number)+".png",crop_img)
            elif valence_value >= VALENCE_CLASS_EDGES[2] and valence_value <= VALENCE_CLASS_EDGES[3]:
                Path(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\valence\\high\\").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\valence\\high\\"+str(frame_number)+".png",crop_img)
            else:
                print("error writing valence image crop")
            #save crop to path based on arousal value
            if arousal_value >= AROUSAL_CLASS_EDGES[0] and arousal_value < AROUSAL_CLASS_EDGES[1]:
                Path(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\arousal\\low\\").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\arousal\\low\\"+str(frame_number)+".png",crop_img)
            elif arousal_value >= AROUSAL_CLASS_EDGES[1] and arousal_value < AROUSAL_CLASS_EDGES[2]:
                Path(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\arousal\\neutral\\").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\arousal\\neutral\\"+str(frame_number)+".png",crop_img)
            elif arousal_value >= AROUSAL_CLASS_EDGES[2] and arousal_value <= AROUSAL_CLASS_EDGES[3]:
                Path(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\arousal\\high\\").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(".\\faces"+"\\"+str(VIDEO_NUMBER)+"\\arousal\\high\\"+str(frame_number)+".png",crop_img)
            else:
                print("error writing arousal image crop")
        

            print(frame_number)
            
        cv2.destroyAllWindows()
        cap.release()





if __name__ == '__main__':
    main()