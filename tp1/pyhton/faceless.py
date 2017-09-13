import numpy as np
import cv2
import os.path
from collections import defaultdict

def load_images(paths,img_nums):
    """
    Load images from a list of sources. Images are loaded in gray scale.

    :param paths: Array of sources to load images.
    :param img_nums: Array of numbers to append to the faces.
    :return: A matrix with each image represented as an array on each row.
    """

    database = []
    classes = {}
    counter = 0
    for path in paths:
        num_files = len([f for f in os.listdir(path)
                         if os.path.isfile(os.path.join(path, f))])
        print "loading %s from %s" %( img_nums, path)
        for x in img_nums:
            classes[counter] = path
            counter = counter+1
            im = cv2.imread("%s/%d.pgm" % (path,x), flags=cv2.IMREAD_GRAYSCALE)
            database.append(np.ravel(im))
    return (np.stack(database),classes)


def calculate_img_weights(eigenfaces_arg, face):
    proyected = []
    for x in range(0, (eigenfaces_arg.shape[0])):
        proyected.append(np.dot(np.transpose(eigenfaces_arg[x]), face))

    proyected = np.stack(proyected)
    return proyected

def norm_img_weights(eigenfaces_arg, face):
    a = calculate_img_weights(eigenfaces_arg,face)
    return a / np.linalg.norm(a, 2)


#normalized_database is the database with the median removed
#face_to_compare is the face with the median removed, the full pixel image
def find_closest_faces(face_to_compare,eigenfaces_arg,normalized_database):
    closestsFaces = []
    face_weights = norm_img_weights(eigenfaces_arg,face_to_compare)
    for x in range(0, (normalized_database.shape[0])):
        val = norm_img_weights(eigenfaces_arg,normalized_database[x]) - face_weights
        val = np.linalg.norm(val, 2)
        closestsFaces.append((x,val))

    closestsFaces = sorted(closestsFaces, key=lambda x: x[1])
    return closestsFaces

def calculate_mean_face(database):
    # Calculating mean of all faces
    return np.int_(database.mean(0))

def calc_eigenfaces(normalized_database,amount):
    # Calculating pseudo-covariance matrix and calculating its eigenvalues
    # and eigenvectors.
    pseudo = np.dot(normalized_database, normalized_database.transpose())
    # print pseudo

    eigenvalues, eigenvectors = np.linalg.eig(pseudo)

    # Calculating eigenfaces using the best eigenvectors (ranked by their
    # eigenvalues). This is made by projecting the original mean-substracted images.
    eigenfaces = []
    for x in range(0, (eigenvectors.shape[0])):
        eigenfaces.append(np.dot(normalized_database.transpose(), eigenvectors[x]))

    return np.stack(eigenfaces)[0:amount]

def analize_matchs(results, classes):
    winner_weights = dict()
    for x in results:
        if (classes[x[0]] in winner_weights):
            winner_weights[classes[x[0]]] = winner_weights[classes[x[0]]] + x[1]
        else:
            winner_weights[classes[x[0]]] = x[1]

    winner_weights = sorted(winner_weights.items(), key=lambda x: x[1])
    return winner_weights

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

