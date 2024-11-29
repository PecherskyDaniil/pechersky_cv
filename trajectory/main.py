import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from skimage.measure import label,regionprops,euler_number
from collections import defaultdict

def distance(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)

arr=[]
filenames = [f for f in listdir("./motion/out/") if isfile(join("./motion/out/", f))]
arr=np.load("./motion/out/h_0.npy")
labeled=label(arr)
regions=regionprops(labeled)
objects=[]
for region in regions:
    cx,cy=region.centroid
    objects.append([[cx,cy]])
for file in range(1,100):
    x=np.load(f"./motion/out/h_{file}.npy")
    regions=regionprops(label(x))
    mins=100000
    mini=0
    ds=[]
    for j in range(len(regions)):
        ds.append([])
        for i in range(len(objects)): 
            ds[j].append(distance(regions[j].centroid,objects[i][-1]))
    ds=np.array(ds)
    objects[0].append(regions[np.argmin(ds[:,0])].centroid)
    ds[np.argmin(ds[:,0])]=100000
    objects[1].append(regions[np.argmin(ds[:,1])].centroid)
    ds[np.argmin(ds[:,1])]=100000
    objects[2].append(regions[np.argmin(ds[:,2])].centroid)
    ds[np.argmin(ds[:,2])]=100000
obs=np.array(objects)
plt.plot(obs[0,:,0],obs[0,:,1])
plt.plot(obs[1,:,0],obs[1,:,1])
plt.plot(obs[2,:,0],obs[2,:,1])
plt.show()

                    
            
        

