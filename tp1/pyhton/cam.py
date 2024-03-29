import pygame
import pygame.camera
import cv2
from kpca import KPCA
from pca import PCA
from sklearn import svm
import numpy as np
from collections import defaultdict
from parameters import *

HEIGHT = 480
WIDTH = 640

HEIGHT_PGM = 112
WIDTH_PGM = 92
names = {0: 'Pedro',
         1: 'Alejo',
         2: "Julian",
         3: "Matias",
         4: "Mariano"}
names = defaultdict(lambda: "Unknown", names)


def rotate_img(img, angle):
    loc = img.get_rect().center
    rot_sprite = pygame.transform.rotate(img, angle)
    rot_sprite.get_rect().center = loc
    return rot_sprite


pygame.camera.init()
pygame.camera.list_cameras()
cam = pygame.camera.Camera("/dev/video0")
cam.start()

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

kpca = KPCA()
db, classes = kpca.get_default_db(Parameters.trnperper)
kpca.set_db(db, classes)

clf = svm.LinearSVC()  # Already implemented SVM
eigenvectors, eig_classes = kpca.calculate_eigenfaces()
clf.fit(eigenvectors, eig_classes)

while True:
    img = cam.get_image()
    img = rotate_img(img, 90)
    img = pygame.surfarray.array3d(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ymin = max(y - int(h * 0.15), 0)
        ymax = min(y + int(h * 1.05), HEIGHT)
        ratio = (ymax - ymin) / float(HEIGHT_PGM)

        x_center = int(x + w / 2)
        xmin = max(int(x_center - (WIDTH_PGM / 2) * ratio), 0)
        xmax = min(int(x_center + (WIDTH_PGM / 2) * ratio), WIDTH)
        face = gray[ymin:ymax, xmin:xmax]  # Why upside down?

        face = cv2.resize(face, (92, 112))

        arr = []
        arr.append(np.ravel(face))
        arr = np.stack(arr)
        class_predicted = clf.predict(kpca.project_images(arr))
        cv2.putText(img, names[class_predicted[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        cv2.putText(img, names[class_predicted[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=1)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print "Closing program..."
        # cv2.imwrite(str(a)+".pgm", face)
        # a += 1
        break

cam.stop()
cv2.destroyAllWindows()
