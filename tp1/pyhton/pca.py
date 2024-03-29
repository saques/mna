from lib import *
import numpy as np
import cv2
from parameters import *

'''
Facial Recognition using KPCA
'''

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92
areasize = IMG_HEIGHT * IMG_WIDTH

personsno = 5


class PCA:

    def __init__(self):
        # Instance Variables
        self.eigenfaces = None
        self.omegas = []
        self.db = None
        self.mean = None
        self.classes = None
        self.trnperper = Parameters.trnperper

    def image_proyection_from_path(self, img_path):
        # Load training and images and their classes
        tstimg = np.ravel(cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE))
        tstimg = tstimg - self.mean

        arr = []
        arr.append(tstimg)
        return self.project_normalized_images(np.stack(arr))

    def project_images(self, imgs):
        return self.project_normalized_images(imgs - self.mean)

    def project_normalized_images(self, imgs):
        omegas_test = []
        for j in range(0, imgs.shape[0]):
            omegas_test.append(calculate_omega(self.eigenfaces, imgs[j]))
        omegas_test = np.stack(omegas_test)

        return omegas_test

    def calculate_eigenfaces(self):
        if self.db is None:
            print "No database set"
            return None

        dbc = self.db - self.mean

        # PseudoCovariance matrix
        pseudo = np.dot(dbc, dbc.transpose())
        ps_eigenvals, ps_eigenvecs = np.linalg.eig(pseudo)

        ps_indexorder = np.argsort(np.absolute(ps_eigenvals))

        sortedV = np.eye(ps_eigenvecs.shape[0])
        for i, x in enumerate(ps_indexorder):
            sortedV[:, i] = ps_eigenvecs[:, x]

        ps_eigenvecs = sortedV

        # Calculating eigenfaces (returning to R^(h*v)
        ef = np.dot(dbc.T, ps_eigenvecs.T).T

        ef = normalize_matrix(ef)

        eigenfaces_alt = []

        for i in xrange(0, personsno):
            eigenfaces_alt.append(ef[i])

        self.eigenfaces = np.stack(eigenfaces_alt)

        ret_classes = []
        self.trnperper = Parameters.trnperper
        for j in range(0, personsno):
            avg = None
            ret_classes.append(self.classes[j * self.trnperper])
            for i in range(0, self.trnperper):
                if i == 0:
                    avg = calculate_omega(self.eigenfaces, dbc[j * self.trnperper + i])
                else:
                    avg += calculate_omega(self.eigenfaces, dbc[j * self.trnperper + i])

            avg = np.divide(avg, self.trnperper)

            self.omegas.append(avg)

        self.omegas = np.stack(self.omegas)
        return self.omegas, ret_classes

    def get_default_db(self, imperper):
        return load_images_and_get_class(imperper,personsno)

    def set_db(self, db, classes):
        self.db = db
        self.mean = np.int_(self.db.mean(0))
        self.classes = classes
