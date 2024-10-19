import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation,binary_erosion,binary_opening,binary_closing
def neighbours2(y,x):
    return (y,x-1),(y-1,x)
def exist(B,nbs):
    left,top=nbs
    if (left[0]>=0 and left[0]<B.shape[0] and
        left[1]>=0 and left[1]<B.shape[1]):
        if B[left]==0:
            left=None
    else:
        left=None
    if (top[0]>=0 and top[0]<B.shape[0] and
        top[1]>=0 and top[1]<B.shape[1]):
        if B[top]==0:
            top=None
    else:
        top=None
    return left,top
def find(label,linked):
    j=label
    while(linked[int(j)]!=0):
        j=linked[int(j)]
    return j
def union(label1,label2,linked):
    j=find(label1,linked)
    k=find(label2,linked)
    if j!=k:
        linked[int(k)]=j
        
def two_pass(B):
    LB=np.zeros_like(B)
    linked=np.zeros(B.size//2 +1,dtype="uint8")
    label=1
    for y in range(B.shape[0]):
        for x in range(B.shape[1]):
            if B[y,x] != 0:
                nbs=neighbours2(y,x)
                existed=exist(B,nbs)
                if existed[0] is None and existed[1] is None:
                    m=label
                    label+=1
                else:
                    lbs=[LB[n] for n in existed if n is not None]
                    m=min(lbs)
                LB[y,x]=m
                for n in existed:
                    if n is not None:
                        lb=LB[n]
                        if lb!=m:
                                union(m,lb,linked)
    for y in range(B.shape[0]):
        for x in range(B.shape[1]):
            if B[y,x] != 0:
                new_label=find(LB[y,x],linked)
                LB[y,x]=new_label
    cb=0
    for c in np.unique(LB):
        LB[LB==c]=cb
        cb+=1
    return LB
if (__name__=="__main__"):
    for w in range(1,7):
        image = np.load("wires/wires"+str(w)+"npy.txt").astype('int8')
        print("wire"+str(w))
        #plt.imshow(image)
        sepimage=two_pass(image)
        struct=np.ones((3,1))
        for c in np.unique(sepimage):
            if c!=0:
                wire=np.zeros_like(sepimage)
                wire=np.logical_or(sepimage==c,wire).astype("int8")
                obrs=two_pass(binary_opening(wire,struct).astype("int8")).max()
                if obrs>0:
                    print("В " +str(c) + " проводе "+str(obrs)+" частей")
                else:
                    print("Провод "+ str(c) + " изодран в хлам")
                #plt.subplot(2,2,c)
                #plt.imshow(two_pass(binary_opening(wire,struct).astype("int8")))
        print("")
        #plt.show()

