from ultralytics import YOLO
from pathlib import Path
import cv2
import time
import numpy as np
path=Path(__file__).parent
model_path=path/ "best.pt"
model=YOLO(model_path)
cv2.namedWindow("Camera",cv2.WINDOW_NORMAL)
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
camera.set(cv2.CAP_PROP_EXPOSURE,-5)
image=cv2.imread("scirock.jpg")

state="idle" #wait,result
prev_time=time.time()
cur_time=time.time()
player1_hand=0
player2_hand=0
timer=5
afk_last_time=0
result_last_time=0
while camera.isOpened():
    ret,frame=camera.read()
    cv2.putText(frame,f"{state}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    results=model(frame)
    #ann_frame=results[0].plot()
    if not results:
        continue
    cur_time=time.time()
    print(len(results[0].boxes.xyxy))
    if len(results[0].boxes.xyxy)==2:
        afk_last_time=time.time()
        result=results[0]
        labels=[]
        for i in range(len(result.boxes.xyxy)):
            labels.append(result.names[result.boxes.cls[i].item()].lower())
        print(labels)
        player1_hand,player2_hand=labels
        if state=="result" and cur_time-result_last_time<5:
            win=-1
            if player1_hand=="paper":
                if  player2_hand=="scissors":
                    win=1
                elif player2_hand=="rock":
                    win=0
            elif player1_hand=="rock":
                if player2_hand=="scissors":
                    win=0
                elif player2_hand=="paper":
                    win=1
            elif player1_hand=="scissors":
                if player2_hand=="rock":
                    win=1
                elif player2_hand=="paper":
                    win=0
            if win<0:
                for i in range(len(result.boxes.xyxy)):
                    x1,y1,x2,y2=result.boxes.xyxy[i].cpu().numpy().astype(int)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(240,255,234),2)
                    #cv2.putText(frame,f"WON",(x1+10,y1+10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(240,255,240),2)
            else:
                x1,y1,x2,y2=result.boxes.xyxy[win].cpu().numpy().astype(int)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"WON",(x1+10,y1+10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
        if state=="result" and cur_time-result_last_time>5:
            state="idle"
            timer=5
        if state=="wait" and timer>0:
            cv2.putText(frame,f"{timer}",(frame.shape[1]//2,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,0),2)
            
            if cur_time-prev_time>=1:
                timer-=1
                prev_time=time.time()
        if state=="wait" and timer==0:
            state="result"
            result_last_time=time.time()
        
        if player1_hand=="rock" and player2_hand=="rock" and state=="idle":
            prev_time=time.time()
            state="wait"
    elif cur_time-afk_last_time>2:
        state="idle"
        timer=5
    cv2.imshow("Camera",frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

