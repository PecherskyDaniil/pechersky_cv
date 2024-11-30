import cv2
import numpy as np

allpencils=0
for i in range(1,13):
    image=cv2.imread(f"./images/img ({i}).jpg")
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower=(0,100,0)
    upper=(255,255,255)
    mask=cv2.inRange(hsv,lower,upper)
    mask=cv2.dilate(mask,None,iterations=2)

    conts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    vit=[]
    for cont in conts:
        (x,y),(h,w,),d=cv2.minAreaRect(cont)
        if max(h,w)>1000 and min(h,w)>60:
            vit.append(max(h,w)/min(h,w))
        else:
            vit.append(0)
    vit=np.array(vit)
    allpencils+=vit[vit>18].shape[0]
    print(f"image{i}: {vit[vit>18].shape[0]} pencils")
print(f"{allpencils} pencils on images")
