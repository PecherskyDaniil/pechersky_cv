from ultralytics import YOLO
from pathlib import Path
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
path=Path(__file__).parent
model_path=path/ "facial_best.pt"
model=YOLO(model_path)
#cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera",cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask",cv2.WINDOW_NORMAL)
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
camera.set(cv2.CAP_PROP_EXPOSURE,-5)
orig_oranges=cv2.imread("oranges.png")
oranges_hsv=cv2.cvtColor(orig_oranges,cv2.COLOR_BGR2HSV)
lower=np.array((14,180,180))
upper=np.array((18,255,255))
oranges_mask=cv2.inRange(oranges_hsv,lower,upper)
kernel=np.array([[0,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,0]],np.uint8)
oranges_mask=cv2.morphologyEx(oranges_mask,cv2.MORPH_CLOSE,kernel,iterations=2)
contours,_=cv2.findContours(oranges_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
sorted_contours=sorted(contours,key=cv2.contourArea)

m=cv2.moments(sorted_contours[-1])
cx=int(m["m10"]/m["m00"])
cy=int(m["m01"]/m["m00"])
bbox_orange=cv2.boundingRect(sorted_contours[-1])
#oranges=cv2.circle(oranges,(cx,cy),5,(255,0,255),5)
#plt.imshow(oranges_hsv[:,:,0])
#plt.show()
#ret,thresh=cv2.threshold(oranges_hsv[:,:,1],100,255,cv2.THRESH_BINARY)
while camera.isOpened():
    ret,frame=camera.read()
    oranges=orig_oranges.copy()
    results=model(frame)
    if len(results)==0 :
        continue
    if results[0].masks is None:
        continue
    masks=results[0].masks
    global_mask=masks[0].data.cpu().numpy()[0,:,:]
    for mask in masks[1:]:
        global_mask+=mask[0].data.cpu().numpy()[0,:,:]
    global_mask=cv2.resize(global_mask,(frame.shape[1],frame.shape[0]))

    rr,cc=draw.disk((5,5),5)
    struct=np.zeros((11,11)).astype(np.uint8)
    struct[rr,cc]=1
    global_mask=cv2.erode(global_mask,struct,iterations=2)
    global_mask=cv2.dilate(global_mask,struct,iterations=3)
    dop_mask=np.zeros_like(frame)
    dop_mask[:,:,0]=global_mask
    dop_mask[:,:,1]=global_mask
    dop_mask[:,:,2]=global_mask
    print(dop_mask.sum(),dop_mask.shape)
    cv2.imshow("Mask",global_mask)
    #print(global_mask.shape)
    annotated=results[0].plot()
    masked_image=(frame*dop_mask).astype(np.uint8)
    x,y,w,h=bbox_orange
    roi=oranges[y:y+h,x:x+w]
    pos=np.where(dop_mask[:,:,0]>0)
    if dop_mask.sum()==0:
        continue
    min_y,max_y=np.min(pos[0]),np.max(pos[0])
    min_x,max_x=np.min(pos[1]),np.max(pos[1])
    masked_image=masked_image[min_y-10:max_y+10,min_x-10:max_x+10]
    dop_mask=dop_mask[min_y-10:max_y+10,min_x-10:max_x+10]
    resized_parts=cv2.resize(masked_image,(bbox_orange[2],bbox_orange[3]))
    resized_mask=cv2.resize(dop_mask[:,:,0],(bbox_orange[2],bbox_orange[3]))*255
    #print(resized_mask.shape,roi.shape)
    bg=cv2.bitwise_and(roi,roi,mask=cv2.bitwise_not(resized_mask))
    combined=cv2.add(bg,resized_parts)
    oranges[y:y+h,x:x+w]=combined
    #cv2.imshow("Image",masked_image)
    cv2.imshow("Camera",oranges)
    
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

