import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops,euler_number
from collections import defaultdict
from pathlib import Path


def recognize(region):
    result="@"
    if region.image.mean()==1:
        result="-"
    else:
        eulnumber=euler_number(region.image,2)
        hole_size=region.area/region.area_filled
        if eulnumber==-1:
            have_v1=np.sum(np.mean(region.image[:,:region.image.shape[1]//2],0)==1)>3
            if have_v1:
                result="B"
            else:
                result="8"
        elif eulnumber==0 and hole_size<0.9:
            image=region.image.copy()
            image[-1,:]=1
            enum=euler_number(image,1)
            
            if enum==-1:
                result='A'
            else:
                have_v1=np.sum(np.mean(region.image[:,:region.image.shape[1]//2],0)==1)>3
                if have_v1:
                    
                    if region.eccentricity<0.6:
                        result="D"
                    else:
                        result="P"
                else:
                    result='0'
        else:
            have_v1=np.sum(np.mean(region.image,0)==1)>3
            if have_v1:
                result="1"
            else:
                if region.eccentricity<0.5:
                    result="*"
                else:
                    image=region.image.copy()
                    image[0,:]=1
                    image[-1,:]=1
                    image[:,0]=1
                    image[:,-1]=1
                    enum=euler_number(image,2)
                    if enum==-1:
                        result="/"
                    elif enum==-3:
                        result="X"
                    else:
                        result="W"
    return result

if (__name__=="__main__"):
    image = plt.imread('symbols.png')[:,:,:3].mean(2)
    image[image>0]=1
    image_labeled=label(image)
    regions_image=regionprops(image_labeled)
    result=defaultdict(lambda:0)
    #path=Path("images")
    #path.mkdir(exist_ok=True)
    #plt.figure()
    for i,region in enumerate(regions_image):
        symbol=recognize(region)
        result[symbol]+=1
        #plt.cla()
        #plt.title(symbol)
        #plt.imshow(region.image)
        #plt.savefig(path/f"image_{i:03d}.png")
    print(result)
    #plt.imshow(image_labeled)
    #plt.show()
