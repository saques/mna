import numpy as np
import cv2
import os.path

def loadImages(paths):
    database = []
    for path in paths:
        num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))])
        print "loading %s from %s" %( num_files, path)
        for x in range(1,num_files+1):
            im = cv2.imread("%s/%d.pgm" % (path,x), flags=cv2.IMREAD_GRAYSCALE)
            database.append(np.ravel(im))
    return np.stack(database)


img_height = 112
img_width = 92

# a = np.arange(36).reshape(6,6)
# a = np.ravel(a)
# b = np.arange(36,72).reshape(6,6)
# b = np.ravel(b)

# loadImages(["../faces/s1"]).shape

# raw_images = []
# for x in range(0, 43):
#     im = cv2.imread("../faces/%d.pgm" % (x), flags=cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread("../faces/%d.pgm" % (x))

    # raw_images.append(np.ravel(im))

    # cv2.imshow('image',im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

db = loadImages(["../faces/s1","../faces/s2","../faces/s3","../faces/s4","../faces/s5"])

print db.shape
mean_face = db.mean(0)
print "RAW MEAN FACE"
print mean_face
# mean_face =  np.ndarray.mean(db,0,np.dtype(np.uint8))
# mean_face =  np.ndarray.mean(db,0)
# mean_face = np.matrix.(db,)
mean_face = np.int_(mean_face)
print "ROUNDED MEAN FACE"
print mean_face
# cv2.imshow('image',np.reshape(mean_face,[112,92]))
# cv2.imshow('image',np.reshape(raw_images[0],[112,92]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
dbc = db - mean_face
print dbc

print "pseudo covariance matrix "
pseudo = np.dot(dbc, dbc.transpose())
print pseudo

print "eigenvaues"
eigenvalues, eigenvectors = np.linalg.eig(pseudo)
print "eigenvaues"
print eigenvalues
print "eigenvectosrz"
print eigenvectors

print (eigenvectors.shape[0])

print "covariance eigenvector calculation (eigenfaces)"
eigenfaces = []
for x in range(0, (eigenvectors.shape[0])):
    eigenfaces.append(np.dot(dbc.transpose(), eigenvectors[x]))

eigenfaces = np.stack(eigenfaces)
print eigenfaces
print eigenfaces.shape


def calculateImgWeights(eigenfaces_arg, face):
    proyected = []
    for x in range(0, (eigenfaces_arg.shape[0])):
        proyected.append(np.dot(np.transpose(eigenfaces_arg[x]), face))

    proyected = np.stack(proyected)
    return proyected

def normImgWeights(eigenfaces_arg, face):
    a = calculateImgWeights(eigenfaces,face)
    return a / np.linalg.norm(a, 2)

def printImg(id):
    im = cv2.imread("../faces/%d.pgm" % (id), flags=cv2.IMREAD_GRAYSCALE)
    cv2.imshow('i%d' % (id),im)
    cv2.resizeWindow('i%d' % (id), 190,150)
    return


def findClosestFace(id,eigenfaces_arg,normalized_database):
    closestsFaces = []
    face_weights = normImgWeights(eigenfaces_arg,normalized_database[id])
    for x in range(0, (normalized_database.shape[0])):
        val = normImgWeights(eigenfaces_arg,normalized_database[x]) - face_weights
        val =  np.linalg.norm(val, 2)
        closestsFaces.append((x,val))

    closestsFaces = sorted(closestsFaces, key=lambda x: x[1])
    return closestsFaces

print "CLOSEST FACES"
print findClosestFace(15,eigenfaces,dbc)

cv2.waitKey(0)
cv2.destroyAllWindows()
