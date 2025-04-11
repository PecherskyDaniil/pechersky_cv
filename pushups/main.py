from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pathlib import Path
import cv2
import time
import numpy as np
path=Path(__file__).parent
model_path=path/ "yolo11n-pose.pt"
model=YOLO(model_path)


def distance(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def angle(a,b,c):
    d=np.rad2deg(np.arctan2(c[1]-b[1],c[0]-b[0]))
    e=np.rad2deg(np.arctan2(a[1]-b[1],a[0]-b[0]))
    angle_=d-e
    angle_=angle_+360 if angle_<0 else angle_
    return 360-angle_ if angle_>180 else angle_

def process_hands(image,keypoints):
    nose_seen=keypoints[0][0]>0 and keypoints[0][1]>0
    left_ear_seen=keypoints[3][0]>0 and keypoints[3][1]>0
    right_ear_seen=keypoints[4][0]>0 and keypoints[4][1]>0
    left_shoulder=keypoints[5]
    right_shoulder=keypoints[6]
    left_elbow=keypoints[7]
    right_elbow=keypoints[8]
    left_hand=keypoints[9]
    right_hand=keypoints[10]
    left_ankle=keypoints[15]
    right_ankle=keypoints[16]
    left_coleno=keypoints[13]
    right_coleno=keypoints[14]
    left_bedro=keypoints[11]
    right_bedro=keypoints[12]
    if left_ear_seen and not(right_ear_seen) and sum(left_shoulder)!=0 and sum(left_elbow)!=0 and sum (left_hand)!=0:
        angle1=angle(left_elbow,left_shoulder,left_bedro)
        angle2=angle(left_ankle,left_coleno,left_bedro)
        x,y=int(left_elbow[0])+10,int(left_elbow[1])+10
        cv2.putText(frame,f"{(angle2):.1f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
        return angle1,angle2
    elif sum(right_shoulder)!=0 and sum(right_elbow)!=0 and sum (right_hand)!=0:
        angle1=angle(right_elbow,right_shoulder,right_bedro)
        angle2=angle(right_ankle,right_coleno,right_bedro)
        x,y=int(right_elbow[0])+10,int(right_elbow[1])+10
        cv2.putText(frame,f"{(angle2):.1f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
        return angle1,angle2
    return None,None
cv2.namedWindow("Camera",cv2.WINDOW_NORMAL)
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
camera.set(cv2.CAP_PROP_EXPOSURE,-4)
last_time=time.time()
flag=False
count=0
writer=cv2.VideoWriter("out.mp4",cv2.VideoWriter.fourcc(*"avc1"),10,(640,480))
last_squat=time.time()
while camera.isOpened():
    ret,frame=camera.read()
    cur_time=time.time()
    cv2.putText(frame,f"{(1/(cur_time-last_time)):.1f}",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
    last_time=time.time()
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    results=model(frame)
    if not results:
        continue
    result=results[0]
    
    keypoints=result.keypoints.xy.tolist()
    if not keypoints:
        continue
    keypoints=keypoints[0]
    if not keypoints:
        continue
    annotator=Annotator(frame)
    annotator.kpts(result.keypoints.data[0],result.orig_shape,5,True)
    annotated=annotator.result()

    angle1_,angle2_=process_hands(annotated,keypoints)
    if time.time()-last_squat>15:
        count=0
    if angle1_ is not None and angle2_ is not None:
        if angle1_<45 and angle2_>165:
            flag=True
        elif flag and (angle1_>45 and angle2_<165):
            count+=1
            last_squat=time.time()
            flag=False
    cv2.putText(frame,f"{count}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
    cv2.imshow("Camera",frame)
    writer.write(frame)
   
model.save(model_path)      
camera.release()
writer.release()
cv2.destroyAllWindows()