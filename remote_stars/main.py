import matplotlib.pyplot as plt
import numpy as np
import socket
from skimage.measure import label,regionprops,euler_number
from collections import defaultdict
host="84.237.21.36"
port=5152

def recvall(sock,n):
    data=bytearray()
    while (len(data)<n):
        packet=sock.recv(n-len(data))
        if not packet:
            return
        data.extend(packet)
    return data

with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
    sock.connect((host,port))
    beat=b'nope'
    while beat!=b'yep':
        sock.send(b'get')
        bts=recvall(sock,40002)
        im1=np.frombuffer(bts[2:],dtype="uint8").reshape(bts[0],bts[1])
        plt.clf()
        plt.subplot(111)
        plt.imshow(im1)
        plt.pause(3)
        bin_im1=(im1>np.unique(im1).mean()).astype("uint8")
        regions=regionprops(label(bin_im1))
        pos1=regions[0].centroid
        pos2=regions[1].centroid
        res=np.abs(np.array(pos1)-np.array(pos2))
        ans=round((res[0]**2+res[1]**2)**0.5,1)
        print(pos1,pos2)
        print(ans)
        sock.send(f"{ans}".encode())
        print(sock.recv(6))
        
        sock.send(b'beat')
        beat=sock.recv(6)
