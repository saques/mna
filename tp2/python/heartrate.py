# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import fft_ct
from scipy.fftpack import fftfreq

cap = cv2.VideoCapture('sample1.mp4')

if not cap.isOpened():
    print("No lo pude abrir")
    exit(0)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1,length))

k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        r[0,k] = np.mean(frame[330:360, 610:640])
    else:
        break
    k = k + 1


cap.release()
cv2.destroyAllWindows()

n = 512
f = (np.linspace(-n/2,n/2-1,n)*fps/n)*60

r = r[0,0:n]-np.mean(r[0,0:n])


R = np.abs(np.fft.fftshift(fft_ct(r)))**2


filter = np.zeros(n)
inc = (fps/n)*60
filter[int(n/2+50/inc):int(n/2+110/inc)] = 1
R *= filter

plt.plot(f,R)
plt.xlim(0,200)
plt.xlabel("freq. [1/min.]")


print ("Heart rate: %f bpm"  % abs(f[np.argmax(R)]))

plt.show()