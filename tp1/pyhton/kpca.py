from lib import *
import numpy as np
import cv2
from sklearn import svm

'''
Facial Recognition using KPCA
'''

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92
areasize = IMG_HEIGHT * IMG_WIDTH

personsno = 5
tstperper = 5
trnperper = 5
tstno = tstperper * personsno
trnno = trnperper * personsno

class KPCA:

    # Class Variables
    degree = 2

    def __init__(self):
        # Instance Variables
        self.eigenvectors = []
        self.K = None
        self.db = None
        self.f = 1

    def image_proyection_from_path(self, img_path):
        # Load training and images and their classes
        tstimg = np.ravel(cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE))

        # Make every pixel have a value between -1 and 1
        tstimg = np.divide(np.add(tstimg, -127.5), 127.5)

        return self.project_normalized_images(tstimg)

    def project_images(self, imgs):
        return self.project_normalized_images(np.divide(np.add(imgs, -127.5), 127.5))

    def project_normalized_images(self, imgs):
        # Project test images
        onesM_tst = np.ones([tstno, trnno]) / trnno
        oneM = np.ones([trnno, trnno]) / trnno
        Ktest = (np.dot(imgs, self.db.T) / trnno + 1) ** self.degree
        Ktest = Ktest - np.dot(onesM_tst, self.K) - np.dot(Ktest, oneM) + np.dot(onesM_tst, np.dot(self.K, oneM))
        return np.dot(Ktest, self.eigenvectors)

    def calculate_eigenfaces(self):
        if self.db is None:
           print "No database set"
           return None

        # Polynomial kernel
        K = (np.dot(self.db, self.db.T) / trnno + 1) ** self.degree
        oneM = np.ones([trnno, trnno]) / trnno
        K = K - np.dot(oneM, K) - np.dot(K, oneM) + np.dot(oneM, np.dot(K, oneM))
        self.K = K

        # Get eigenvectors and sort them by their associated eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(K)   # sorted in ascending order
        eigenvalues = np.flipud(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)

        # Normalize eigenvectors
        for col in range(eigenvectors.shape[1]):
            eigenvectors[:, col] = eigenvectors[:, col] / np.sqrt(np.abs(eigenvalues[col]+0.0000001))
        self.eigenvectors = eigenvectors

        # Project training images
        return np.dot(K.T, eigenvectors)

    def get_default_db(self, imperper):
        return load_images_and_get_class(imperper, personsno)

    def set_db(self, db):
        self.db = np.divide(np.add(db, -127.5), 127.5)
