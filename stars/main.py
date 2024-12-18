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
    image = np.load("stars.npy.txt")#.astype('int8')
    plt.subplot(2,2,1)
    plt.imshow(image)
    disstruct=np.ones((1,2))
    disimage=binary_dilation(image,disstruct).astype("uint16")
    allfg=two_pass(disimage).max()
    unstars=np.ones((2,2))
    unstars=binary_erosion(image,unstars).astype("uint16")
    plt.subplot(2,2,2)
    plt.imshow(unstars)
    print(allfg-two_pass(unstars).max())
    plt.show()

