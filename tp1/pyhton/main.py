from lib import *
import numpy as np
import cv2
import os.path

'''
Facial Recognition using PCA
'''


def load_images(paths):
    """
    Load images from a list of sources. Images are loaded in gray scale.

    :param paths: Array of sources to load images.
    :return: A matrix with each image represented as an array on each row.
    """

    database = []
    for path in paths:
        num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))])
        print "loading %s from %s" %( num_files, path)
        for x in range(1,num_files+1):
            im = cv2.imread("%s/%d.pgm" % (path,x), flags=cv2.IMREAD_GRAYSCALE)
            database.append(np.ravel(im))
    return np.stack(database)


def print_image(mat, name='image', time=1000):
    """
    Shows an image in a separate windows

    :param mat: Matrix where the image is stored.
                Its values must be between 0 and 255.
    :param name: Name of the image's window.
    :param time: Time the images will be display (0 is no limit).
    """

    cv2.imshow(name, mat / 255.)
    cv2.waitKey(time)
    cv2.destroyAllWindows()

def print_images(mats, name='image', time=1000):
    """
    Shows images in a separate windows

    :param mats: Matrices where the images are stored.
                 Its values must be between 0 and 255.
    :param name: Name of the image's window.
    :param time: Time the images will be display (0 is no limit).
    """

    for index, m in enumerate(mats):
        cv2.imshow(name + str(index), m / 255.)

    cv2.waitKey(time)
    cv2.destroyAllWindows()


# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92

trnperper = 5
trnno = trnperper*40  # training number

# Load all images from database (located in '../faces')
# db = loadImages(["../faces/s1","../faces/s2","../faces/s3","../faces/s4","../faces/s5"])
# db = load_images(["../faces"])
db = load_images_2(trnperper)

# Calculating mean of all faces
mean_face = db.mean(0)
mean_face = np.int_(mean_face)
dbc = db - mean_face

print_image(np.reshape(mean_face, [112, 92]), "mean", 500)


# Calculating pseudo-covariance matrix and calculating its eigenvalues
# and eigenvectors.
# pseudo = np.dot(dbc, dbc.transpose())
# eigenvalues, eigenvectors = np.linalg.eig(pseudo)


# KERNEL: polinomial de grado degree
degree = 2
K = (np.dot(dbc, dbc.T) / trnno + 1) ** degree
# K = (K + K.T)/2.0

# esta transformacion es equivalente a centrar las imagenes originales...
unoM = np.ones([trnno, trnno]) / trnno
K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))

eigenvalues, eigenvectors = np.linalg.eigh(K)
eigenvalues = np.flipud(eigenvalues)
eigenvectors = np.fliplr(eigenvectors)

for col in range(eigenvectors.shape[1]):
    eigenvectors[:,col] = eigenvectors[:,col]/np.sqrt(np.abs(eigenvalues[col]))


# Calculating eigenfaces using the best eigenvectors (ranked by their
# eigenvalues). This is made by projecting the original mean-substracted images.
# eigenfaces = []
# for x in range(eigenvectors.shape[0]):
#     eigenfaces.append(np.dot(dbc.transpose(), eigenvectors[x]))
# eigenfaces = np.stack(eigenfaces)

eigenfaces = np.dot(K.T, eigenvectors)

def find_closest_face(id, eigenfaces_arg, normalized_database):
    closest_faces = []
    face_weights = norm_img_weights(eigenfaces_arg, normalized_database[id])
    for x in range(normalized_database.shape[0]):
        val = norm_img_weights(eigenfaces_arg, normalized_database[x]) - face_weights
        val = np.linalg.norm(val, 2)
        closest_faces.append((x, val))

    closest_faces = sorted(closest_faces, key=lambda x: x[1])
    return closest_faces


def norm_img_weights(eigenfaces_arg, face):
    a = calculate_img_weights(eigenfaces_arg, face)
    return a / np.linalg.norm(a, 2)


def calculate_img_weights(eigenfaces_arg, face):
    projected = []
    for x in range(eigenfaces_arg.shape[0]):
        projected.append(np.dot(np.transpose(eigenfaces_arg[x]), face))

    projected = np.stack(projected)
    return projected

def find_closest_face(proyected_face, eigenfaces_arg, normalized_database):
    closest_faces = []
    face_weights = norm_img_weights(eigenfaces_arg, proyected_face)
    for x in range(normalized_database.shape[0]):
        val = norm_img_weights(eigenfaces_arg, normalized_database[x]) - face_weights
        val = np.linalg.norm(val, 2)
        closest_faces.append((x, val))

    closest_faces = sorted(closest_faces, key=lambda x: x[1])
    return closest_faces


unoML = np.ones([1, trnno]) / trnno
Ktest = (np.dot(dbc[43], dbc.T) / trnno + 1) ** degree
Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
imtstproypre = np.dot(Ktest, eigenvectors)


print eigenfaces.shape
print imtstproypre

# print find_closest_face(imtstproypre, eigenfaces, db)

# Look for the nearest face
# print "CLOSEST FACES"
# closest_faces = find_closest_face(43, eigenfaces, dbc)
# print closest_faces
#
# print_images([np.reshape(db[closest_faces[1][0]],[112,92]),
#               np.reshape(db[43], [112, 92])], time=0)

