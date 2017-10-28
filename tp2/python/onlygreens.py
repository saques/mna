

import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import fft_ct


cap = cv2.VideoCapture('sample6.mp4')

if not cap.isOpened():
    print("No lo pude abrir")
    exit(0)



length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print width
print height

fps = cap.get(cv2.CAP_PROP_FPS)

NAME = 'sample5og.avi'
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(NAME, fourcc, fps, (width,height))


k = 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame[:, :, 0] -= frame[:, :, 0]
        #frame[:, :, 1] -= frame[:, :, 1]
        frame[:, :, 2] -= frame[:, :, 2]
        out.write(frame)
    else:
        break
    k = k + 1

cap.release()
out.release()
cv2.destroyAllWindows()










