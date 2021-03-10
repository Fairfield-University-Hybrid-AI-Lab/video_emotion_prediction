import numpy as np
from cv2 import cv2
import os.path
import landmarkoutline
import CropMaskOverlay
from pathlib import Path
import datetime
import uuid

VIDEO_NUMBER = 202
FOLDER_PATH = "\\Users\\George\\Documents\\Python\\ADS CapStone\\aff_wild_annotations_bboxes_landmarks_new"
BOUNDING_BOX_PATH = "\\Users\\George\\Documents\\Python\\ADS CapStone\\aff_wild_annotations_bboxes_landmarks_new\\landmarks\\train"
ANNOTATIONS_PATH =  "\\Users\\George\\Documents\\Python\\ADS CapStone\\aff_wild_annotations_bboxes_landmarks_new\\annotations\\train"

def main():
    #load annotations
    valence_annotations = landmarkoutline.load_annotations(ANNOTATIONS_PATH +"\\valence\\"+ str(VIDEO_NUMBER)+".txt")
    arousal_annotations = landmarkoutline.load_annotations(ANNOTATIONS_PATH +"\\arousal\\"+ str(VIDEO_NUMBER)+".txt")

    frame_number = 0
    

    cap = cv2.VideoCapture("\\Users\\George\\Documents\\Python\\ADS CapStone\\aff_wild_annotations_bboxes_landmarks_new\\videos\\train\\"+str(VIDEO_NUMBER)+".mp4")
    
    
    while(cap.isOpened()):
        frame_number += 1

        #load bounding box coords
        bounding_box = landmarkoutline.load_points(BOUNDING_BOX_PATH +"\\"+ str(VIDEO_NUMBER)+"\\"+str(frame_number)+".pts")
        if not bounding_box:
            if frame_number > 10000:
                break
            print(VIDEO_NUMBER,"Failed to retrieve BB Points", frame_number)
            continue
        ret, frame = cap.read()
        
        if ret == False:
            print(VIDEO_NUMBER,"Failed to retrieve ret")
            break 
        
        if frame is None:
            print(VIDEO_NUMBER,"Failed to retrieve frame")
            break 
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pts = np.array(bounding_box[0:27], np.int32)
        pts = pts.reshape((-1,1,2))
        
        
        
        dst = CropMaskOverlay.CROPMASK(gray,pts,155)
        crop_img = dst
        cv2.imshow("cropped", crop_img)


        
        try:
            valence_value = float(valence_annotations[frame_number])
        except:
            print(VIDEO_NUMBER, "Broke via valence value index error")
            break
        
        try:
            arousal_value = float(arousal_annotations[frame_number])
        except:
            print(VIDEO_NUMBER, "Broke via arousal value index error")
            break
        
        
        #save crop to path based on valence value
        if valence_value >= -1 and valence_value < -0.5:
            Path(FOLDER_PATH+"\\landmarkimg"+"\\valence\\low\\").mkdir(parents=True, exist_ok=True)#create directory path if it doesnt exist
            cv2.imwrite(FOLDER_PATH+"\\landmarkimg"+"\\valence\\low\\"+str(uuid.uuid4())+".png",crop_img)
       
        elif valence_value >= -0.5 and valence_value < 0.5:
            Path(FOLDER_PATH+"\\landmarkimg"+"\\valence\\neutral\\").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(FOLDER_PATH+"\\landmarkimg"+"\\valence\\neutral\\"+str(uuid.uuid4())+".png",crop_img)
       
        elif valence_value >= 0.5 and valence_value <= 1:
            Path(FOLDER_PATH+"\\landmarkimg"+"\\valence\\high\\").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(FOLDER_PATH+"\\landmarkimg"+"\\valence\\high\\"+str(uuid.uuid4())+".png",crop_img)
        else:
            print("error writing valence image crop")
        
        #save crop to path based on arousal value
        if arousal_value >= -1 and arousal_value < -0.5:
            Path(FOLDER_PATH+"\\landmarkimg"+"\\arousal\\low\\").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(FOLDER_PATH+"\\landmarkimg"+"\\arousal\\low\\"+str(uuid.uuid4())+".png",crop_img)
       
        elif arousal_value >= -0.5 and arousal_value < 0.5:
            Path(FOLDER_PATH+"\\landmarkimg"+"\\arousal\\neutral\\").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(FOLDER_PATH+"\\landmarkimg"+"\\arousal\\neutral\\"+str(uuid.uuid4())+".png",crop_img)
       
        elif arousal_value >= 0.5 and arousal_value <= 1:
            Path(FOLDER_PATH+"\\landmarkimg"+"\\arousal\\high\\").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(FOLDER_PATH+"\\landmarkimg"+"\\arousal\\high\\"+str(uuid.uuid4())+".png",crop_img)
       
        else:
            print("error writing arousal image crop")
     
       
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


while VIDEO_NUMBER < 450:
    main()
    print(VIDEO_NUMBER, "Completed at", datetime.datetime.now())
    VIDEO_NUMBER = VIDEO_NUMBER+1
    

# if __name__ == '__main__':
#     main()