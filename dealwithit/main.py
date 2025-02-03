import cv2
import matplotlib.pyplot as plt
import numpy as np

lbp_cascade="lbpcascades/lbpcascade_frontalface.xml"
haar_cascade1="haarcascades/haarcascade_eye.xml"
haar_cascade2="haarcascades/haarcascade_eye_tree_eyeglasses.xml"
eyes1=cv2.CascadeClassifier(haar_cascade1)
eyes2=cv2.CascadeClassifier(haar_cascade2)
#lbp=cv2.CascadeClassifier(lbp_cascade)


def detector(img,classifier,scaleFactor=None,minNeighbours=None):
    result=img.copy()
    rects=classifier.detectMultiScale(result,scaleFactor=scaleFactor,minNeighbors=minNeighbours)
    for (x,y,h,w) in rects:
        cv2.rectangle(result,(x,y),(x+w,y+h),(0,128,255))
    #result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def add_glasses(glasses,img,classifier,scaleFactor=None,minNeighbours=None):
    result=img.copy()
    rects=classifier.detectMultiScale(result,scaleFactor=scaleFactor,minNeighbors=minNeighbours)
    if len(rects)==2:
        xc=int((rects[0][0]+rects[1][0]+rects[1][3])/2)
        yc=int((rects[0][1]+rects[1][1]+rects[1][2])/2)
        wg=max(rects[1][0]+rects[1][3]-rects[0][0],rects[0][0]+rects[0][3]-rects[1][0])
        hg=max(rects[1][1]+rects[1][2]-rects[0][1],rects[0][1]+rects[0][2]-rects[1][1])
        gl=glasses.copy()
        gl=cv2.resize(gl, (wg+50,hg))
        for i in range(yc-int(hg/2),yc-int(hg/2)+gl.shape[0]):
            for j in range(xc-int(wg/2)-25,xc-int(wg/2)-25+gl.shape[1]):
                if np.all(gl[i-yc-int(hg/2),j-xc-int(wg/2)-25]<(250,250,250)):
                    result[i,j]=gl[i-yc-int(hg/2),j-xc-int(wg/2)-25]
    #for (x,y,h,w) in rects:
    #    cv2.rectangle(result,(x,y),(x+w,y+h),(0,128,255))
    #result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

#sheldone=cv2.imread("cooper.jpg")
#solvay=cv2.imread("solvay-conference.jpg")

#plt.figure()

#plt.imshow(detector(sheldone,face,1.2,5))
#plt.figure()

#plt.imshow(detector(solvay,lbp,1.2,5))
#plt.show()

cv2.namedWindow("Camera",cv2.WINDOW_NORMAL)
    
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
camera.set(cv2.CAP_PROP_EXPOSURE,-3)
glasses=cv2.imread("dealwithit.png")
gray_glasses=cv2.cvtColor(glasses,cv2.COLOR_BGR2GRAY)
gray_glasses=255-gray_glasses
conts,_=cv2.findContours(gray_glasses,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(conts[0])
glasses=glasses[y:y+h,x:x+w]
while camera.isOpened():
    ret,frame=camera.read()
    
    cv2.imshow("Camera",add_glasses(glasses,frame,eyes2,1.2,5))
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()
