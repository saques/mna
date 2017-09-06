import numpy as np
import cv2

img_height = 112
img_width = 92

# a = np.arange(36).reshape(6,6)
# a = np.ravel(a)
# b = np.arange(36,72).reshape(6,6)
# b = np.ravel(b)

raw_images = []
for x in range(0, 5):
    im = cv2.imread("../faces/%d.pgm" % (x),flags=cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread("../faces/%d.pgm" % (x))

    raw_images.append(np.ravel(im))

    # cv2.imshow('image',im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


db = np.stack(raw_images)
print db
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
dbc = db-mean_face
print dbc

print "pseudo covariance matrix "
pseudo = np.dot(dbc, dbc.transpose())
print pseudo

print "eigenvaues"
eigenvalues,eigenvectors = np.linalg.eig(pseudo)
print "eigenvaues"
print eigenvalues
print "eigenvectosrz"
print eigenvectors

print (eigenvectors.shape[0])

print "covariance eigenvector calculation (eigenfaces)"
eigenfaces = []
for x in range(0, (eigenvectors.shape[0])):
     eigenfaces.append(np.dot(dbc.transpose(), eigenvectors[x]))

eigenfaces= np.stack(eigenfaces)
print eigenfaces
print eigenfaces.shape

def calculateImgWeights( eigenfaces_arg,face):
    proyected = []
    for x in range(0, (eigenfaces_arg.shape[0])):
        proyected.append(np.dot(np.transpose(eigenfaces_arg[x]),face))

    proyected = np.stack(proyected)
    return proyected


print calculateImgWeights(eigenfaces,dbc[0])
