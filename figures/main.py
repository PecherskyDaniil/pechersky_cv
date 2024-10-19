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
    linked=np.zeros(B.size//2 +1,dtype="uint16")
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
    image = np.load("ps.npy.txt").astype('uint16')
    splitedimage=two_pass(image)
    print("all figures:",splitedimage.max())
    structsg=np.ones((3,4,6)).astype("uint16")
    structsv=np.ones((3,6,4)).astype("uint16")
    b0=0
    for h in range(0,2):
        structsg[h+1,h*2:h*2+2,2:4]=0
        structsv[h+1,2:4,h*2:h*2+2]=0
    for i in range(structsg.shape[0]):
        pim=two_pass(binary_erosion(image,structsg[i,:,:]).astype("uint16"))
        print(structsg[i,:,:],"\n"+str(pim.max()-b0)+"\n")
        if i==0:
            b0=pim.max()
    b0=0
    for i in range(structsv.shape[0]):
        pim=two_pass(binary_erosion(image,structsv[i,:,:]).astype("uint16"))
        print(structsv[i,:,:],"\n"+str(pim.max()-b0)+"\n")
        if i==0:
            b0=pim.max()

