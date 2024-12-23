import cv2
import numpy
import time
import mss
import matplotlib.pyplot as plt
import pyautogui
foundDino=False
#cv2.namedWindow("Monitor",cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Monitor",600, 60)
sct = mss.mss()

pyautogui.click("dinocolor.png")
h,w=pyautogui.size()
x,y=pyautogui.position()
pyautogui.write(' ', interval=0.1)
#time.sleep(2)
#pyautogui.write(' ', interval=0.3)
starttime=time.time()
last_time=time.time_ns() // 1_000_000
time_interval=1000000000  
x+=3
y+=8
swidth=100
sheight=40
sx=80
tm=-10
dc=0
while True:
    #break
    img =numpy.asarray(sct.grab(sct.monitors[1]))[y-60:y+30,x-30:x+719]
    gray=cv2.cvtColor(img[:sheight+30,30:],cv2.COLOR_BGR2GRAY)
    thresh=cv2.threshold(gray,220,255,cv2.THRESH_BINARY)[1]
    mask=cv2.bitwise_not(thresh[30:,49:500])
    mask=cv2.dilate(mask,None,iterations=6)
    #cv2.rectangle(img, (227+int(tm),0),(227+int(tm),90), (0,0,255), 2)
    #cv2.imshow("Monitor", mask)
    conts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if thresh[:,320:330].mean()<250:
        #if time.time()-starttime<58:
        tm+=1.26+0.3*(time.time()-starttime>8)+1*(time.time()-starttime>20)+0.54*(time.time()-starttime>50)-1.99*(tm>245)-0.8*(tm>251)-1*(tm>265)
        print(tm)
    tm-=(0.026-0.006*(tm>250))#(0.02+0.015*((time.time()-starttime)//50))
    #if thresh[:,int(0.5*((time.time()-starttime)/9)):200].mean()==255 and dc:
    #    pyautogui.keyDown("down")
        #pyautogui.sleep(0.01)
    #    dc=0
    if dc and thresh[20:35,0:180].mean()>252:
        pyautogui.keyUp("down")
        dc=0
    for cont in conts:
        (cacx, cacy, cacw, cach)=cv2.boundingRect(cont)
        if cacx+cacw+cach*0.1<140+int(tm):
            if (cacy+cach)<36:
                pyautogui.keyDown("down")
                pyautogui.sleep(0.3)
                dc=1
            else:
                pyautogui.write(" ")
                #dc=1
                        #else:
            #    tm=400
            
    
    #img=cv2.rectangle(img, (sx+int(tm),0),(sx+swidth+int(tm),sheight), (0,255,0), 2)
    #img=cv2.rectangle(img, (sx,0),(sx+swidth,sheight), (255,0,0), 2)    #Display the picture
    #cv2.putText(img,f"Timer = {time_interval-(time.time_ns() // 1_000_000) -last_time} s",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))
    
    
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()

