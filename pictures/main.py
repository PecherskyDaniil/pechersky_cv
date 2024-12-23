import cv2
import numpy as np
import os
#cv2.namedWindow("Image",cv2.WINDOW_GUI_NORMAL)
capture=cap = cv2.VideoCapture('output.avi')
c=0
ret,frame=capture.read()
while frame is not None:
    image=frame
    blurred=cv2.GaussianBlur(image,(9,9),0)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    ret,thresh1=cv2.threshold(hsv[:,:,1],100,255,cv2.THRESH_BINARY)
    ret,thresh2=cv2.threshold(hsv[:,:,2],80,255,cv2.THRESH_BINARY)
    contours1,_=cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours2,_=cv2.findContours(thresh2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("Image",frame)
    if len(contours1)==3 and len(contours2)==1:
        c+=1
        print(c)
        #key=cv2.waitKey(100)
    #else:
        #key=cv2.waitKey(1)
    #if key==ord('q'):
    #    break
    ret,frame=capture.read()
#cv2.destroyAllWindows()
print("Число моих изображений(pechersky_da.png)",c)
