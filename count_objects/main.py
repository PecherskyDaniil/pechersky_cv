import cv2
import zmq
import numpy as np

context=zmq.Context()

socket=context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE,b"")
position=None
port=5555
socket.connect("tcp://192.168.0.100:%s" % port)
cv2.namedWindow("Client recv",cv2.WINDOW_GUI_NORMAL)
#cv2.namedWindow("Mask",cv2.WINDOW_NORMAL)
count=0
lower=(10,0,0)
upper=(170,255,255)

flimit=100
slimit=200
def fupdate(value):
    global flimit
    flimit=value
    
def supdate(value):
    global slimit
    slimit=value

#cv2.createTrackbar("F","Client recv",flimit,255,fupdate)
#cv2.createTrackbar("S","Client recv",slimit,255,supdate)

while True:
    figs=0
    cc=0
    cs=0
    msg=socket.recv()
    frame=cv2.imdecode(np.frombuffer(msg,np.uint8),-1)
    count+=1
    blurred=cv2.GaussianBlur(frame,(9,9),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    #threshsv=cv2.inRange(hsv,(0,0,0),(255,255,255))
    gray=cv2.add(cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY),hsv[:,:,1])
    th, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(bw, 190, 50)
    #edges=cv2.dilate(edges,None,iterations=1)
    conts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cont in conts:
        (x,y),r=cv2.minEnclosingCircle(cont)
        
        if r>35:
            #cv2.drawContours(frame,[cont],0,(0,0,0))
            rect=cv2.minAreaRect(cont)
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            cv2.drawContours(frame,[box],0,(0,255,0))
            #print(cv2.contourArea(cont)/cv2.contourArea(box))
            if cv2.contourArea(cont)/cv2.contourArea(box)>0.9:
                cs+=1
            else:
                cc+=1
            figs+=1
    
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    #cv2.putText(frame,f"COunt {count}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
    cv2.putText(frame,f"Figs count {figs}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
    cv2.putText(frame,f"Circles count {cc}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
    cv2.putText(frame,f"Squares count {cs}",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
    cv2.imshow("Client recv",frame)
    #cv2.imshow("Mask",bw)
cv2.destroyAllWindows()

