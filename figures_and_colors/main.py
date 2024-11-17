import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops,euler_number
from collections import defaultdict
from pathlib import Path
from skimage.filters  import threshold_otsu,sobel
from skimage.segmentation import flood_fill
from skimage.color import rgb2hsv

if (__name__=="__main__"):
    image = plt.imread('./balls-and-rects/balls_and_rects.png')
    binary=image.mean(2)
    binary[binary>0]=1
    labeled=label(binary)
    regions=regionprops(labeled)
    image_hsv=rgb2hsv(image)
    colors=[]
    for region in regions:
        cy,cx=region.centroid
        colors.append(image_hsv[int(cy),int(cx)][0])
    colors=np.array(sorted(colors))
    cnum=1
    p=(colors[1:]-colors[:-1]).std()
    colorssp=[]
    for i in range(1,colors.shape[0]):
        if colors[i]-colors[i-1]>p:
            colorssp.append(colors[i])
            cnum+=1
    colorcirc=np.zeros_like(colorssp)
    colorrect=np.zeros_like(colorssp)
    colorall=np.zeros_like(colorssp)
    for region in regions:
        cy,cx=region.centroid
        for i in range(len(colorssp)):
            if image_hsv[int(cy),int(cx)][0]-colorssp[i]<=p:
                colorall[i]+=1
                if region.area/(region.image.shape[0]*region.image.shape[1])==1:
                    colorrect[i]+=1
                else:
                    colorcirc[i]+=1
                break
    print(f"Количество квадратов по цветам: {colorrect}\nКоличество кругов по цветам: {colorcirc}\nКоличество всех фигур по цветам: {colorall}")
