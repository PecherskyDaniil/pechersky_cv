import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label,regionprops
from collections import defaultdict
from pathlib import Path
def extractor(region):
    area=(np.sum(region.image)/(region.image.size))
    perimeter=(region.perimeter/(region.image.size))
    wh=np.sum(region.image[:int(region.image.shape[0]/2)])/np.sum(region.image[int(region.image.shape[0]/2):])
    cy,cx=region.local_centroid
    cy/=region.image.shape[0]
    cx/=region.image.shape[1]
    euler=region.euler_number
    px=int(region.image.shape[0]/5)
    py=int(region.image.shape[1]/5)
    kl=3*np.sum(region.image[px:-px,py:-py])/(region.image.size)
    px=int(region.image.shape[0]*0.45)
    py=int(region.image.shape[1]*0.45)
    kls=2*np.sum(region.image[px:-px,py:-py])/(region.image.size)
    pm=(region.image.shape[0]/region.image.shape[1])
    eccentricity=region.eccentricity*8
    have_v1=(np.sum(np.mean(region.image,0)>0.87)>2)*3
    have_g1=(np.sum(np.mean(region.image,1)>0.85)>2)*4
    have_g2=(np.sum(np.mean(region.image,1)>0.5)>2)*5
    hole_size=(np.sum(region.image)/region.filled_area)
    solidity=region.solidity*2
    ans=np.array([area,perimeter,cy,cx,euler,eccentricity,have_v1,hole_size,have_g1,have_g2,kl,pm,kls,solidity,wh])
    return ans

def classificator(region,classes):
    def_class=None
    s=extractor(region)
    min_d=10**10
    for cls in classes:
        d=distance(s,classes[cls])
        if d<min_d:
            def_class=cls
            min_d=d
    return def_class

labels=os.listdir("task/train/")
x=[]
y=[]
for labelind in range(len(labels)):
    for filename in os.listdir("task/train/"+labels[labelind]):
        template=plt.imread("task/train/"+labels[labelind]+"/"+filename)[:,:,:3].mean(2)
        template[template>0]=1
        template_labeled=label(template)
        regions=regionprops(template_labeled)
        if np.sum(regions[0].image)>250:
            x.append(extractor(regions[0]))
        else:
            x.append(extractor(regions[1]))
        y.append(labelind)
x=np.array(x)
y=np.array(y)
knn=cv2.ml.KNearest_create()

train=x.astype("f4")
responses=y.reshape(-1,1).astype("f4")

knn.train(train,cv2.ml.ROW_SAMPLE,responses)
for test in os.listdir("task/"):
    if test!="train":
        print(test)
        template=plt.imread("task/"+test)[:,:,:3].mean(2)
        template[template<0.1]=0
        template[template>0]=1
        template_labeled=label(template)
        #plt.imshow(template_labeled)
        #plt.show()
        regions=regionprops(template_labeled)
        regions=sorted(regions, key=lambda x: x.centroid[1])
        text=[]
        for i in range(len(regions)):
            if np.sum(regions[i].image)>250:
                new_point=extractor(regions[i])
                ret,results,neighbours,dist= knn.findNearest(np.array(new_point).astype("f4").reshape(1,len(new_point)),2)
                a=regions[i].bbox
                if i!=0 and a[1]-lasta[-1]>30:
                    print(" ",end="")
                lasta=a
                print(labels[int(ret)][-1],end="")
        print("\n")
