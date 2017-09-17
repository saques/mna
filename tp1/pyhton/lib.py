import time
from math import *
from scipy.linalg import hessenberg as hess
import numpy as np
import cv2


NUM_INDIVIDUALS = 40
TOLERANCE = 1E-4


# Returns a matrix containing NUM_INDIVIDUAL rows
# of each picture named "id.pgm"
def load_images(i):
    database = []
    for path in xrange(0, NUM_INDIVIDUALS):
        im = cv2.imread("../orl_faces/s%d/%d.pgm" % (path+1, i), flags=cv2.IMREAD_GRAYSCALE)
        database.append(np.ravel(im))
    return np.stack(database)


def load_images_2(i):
    database = []
    for path in xrange(0, NUM_INDIVIDUALS):
        for x in xrange(0,i):
            im = cv2.imread("../orl_faces/s%d/%d.pgm" % (path+1, x+1), flags=cv2.IMREAD_GRAYSCALE)
            database.append(np.ravel(im))
    return np.stack(database)


def load_images_3(i, j):
    database = []
    for path in xrange(0, j):
        for x in xrange(0,i):
            im = cv2.imread("../orl_faces/s%d/%d.pgm" % (path+1, x+1), flags=cv2.IMREAD_GRAYSCALE)
            database.append(np.ravel(im))
    return np.stack(database)


def load_images_4(i, j):
    database = []
    for path in xrange(0, j):
        im = cv2.imread("../orl_faces/s%d/%d.pgm" % (path+1, i), flags=cv2.IMREAD_GRAYSCALE)
        database.append(np.ravel(im))
    return np.stack(database)



# Returns a vector with Vi = weight of the i-th
# eigenface relative to the face passed as parameter
def calculate_omega(eigfaces, face):
    projected = []
    for x in range(0, (eigfaces.shape[0])):
        projected.append(np.dot(eigfaces[x].transpose(), face))
    projected = np.stack(projected)
    return projected


def normalize_matrix(m):
    ans = []
    for p in range(0, m.shape[0]):
        ans.append(m[p, :] / np.linalg.norm(m[p, :]))
    ans = np.stack(ans)
    return ans


def wilkinson(a, b, c):
    delta = (a-c)/2
    return c - np.sign(delta)*b**2/(np.abs(delta)+np.sqrt(delta**2 + b**2))


def eig(a):
    p, x = hessemberg(a)
    d = dim = x.shape[0]
    values = []
    vectors = np.eye(d)
    for i in range(0, d):
        identity = np.eye(dim)
        prev = None

        while prev is None or np.abs(prev-x[dim-1, dim-1]) > TOLERANCE:
            prev = x[dim-1, dim-1]
            mu = wilkinson(x[dim-2, dim-2], x[dim-2, dim-1], x[dim-1, dim-1])
            q, r = qr_householder(x[0:dim, 0:dim] - np.dot(mu, identity))

            qi = np.eye(d)
            qi[0:dim, 0:dim] = q
            vectors = np.dot(vectors, qi)

            x[0:dim, 0:dim] = np.dot(r, q) + mu*identity

        values.append(x[dim-1, dim-1])
        dim -= 1
    return np.stack(values).round(-1*int(np.log10(TOLERANCE))), vectors


def givens_rotation(a, b):
    if b == 0:
        return 1, 0
    else:
        if abs(b) > abs(a):
            r = float(a) / b
            s = 1 / sqrt(1 + pow(r, 2))
            c = s*r
            return c, s
        else:
            r = float(b) / a
            c = 1 / sqrt(1 + pow(r, 2))
            s = c*r
            return c, s


def qr_givens(a):
    m, n = np.shape(a)
    Q = np.eye(m)
    R = np.copy(a)
    for y in range(0, m):
        for x in reversed(range(y+1, m)):
            if R[x, y] != 0:
                c, s = givens_rotation(R[x-1, y], R[x, y])
                G = np.eye(m)
                sq = np.array([[c, -s], [s, c]])
                G[x-1:x+1, x-1:x+1] = sq
                R = np.dot(G.transpose(), R)
                Q = np.dot(Q, G)
    return Q, R


def v_householder(x):
    v = []
    alpha = np.linalg.norm(x)
    alpha *= -1*np.sign(x[0])
    v0 = -1*np.sign(alpha)*sqrt((alpha-x[0])/(2*alpha))
    v.append(v0)
    for i in range(1, np.shape(x)[0]):
        v.append((-1*x[i])/(2*alpha*v0))
    return np.stack(v)


# Better complexity
def ov_householder(x):
    norm = np.linalg.norm(x)
    u = x.astype(float)
    u[0] += np.sign(u[0])*norm
    v = np.divide(u, np.linalg.norm(u))
    return v


def qr_householder(a):
    m, n = np.shape(a)
    Q = np.eye(m)
    R = a.astype(float)
    for i in range(0, n-1):
        vh = ov_householder(R[i:, i]).reshape((m-i, 1))
        H = np.eye(m)
        H[i:, i:] -= 2.0 * np.dot(vh, vh.transpose())
        Q = np.dot(Q, H)
        R = np.dot(H, R)
    return Q, R


def hessemberg(a):
    m, n = np.shape(a)
    P = np.eye(m)
    R = a.astype(float)
    for i in range(0, n-2):
        vh = ov_householder(R[i+1:, i]).reshape((m-i-1, 1))
        H = np.eye(m)
        H[i+1:, i+1:] -= 2.0 * np.dot(vh, vh.transpose())
        P = np.dot(H, P)
        R = np.dot(H, R)
        R = np.dot(R, H.transpose())
    return P, R


def eigenmatrix(dim):
    i = np.eye(dim)
    for x in range(0, dim):
        i[x, x] = x+1
    r = np.random.rand(dim, dim)
    return np.dot(r, np.dot(i, np.linalg.inv(r)))

A = eigenmatrix(10)

values, vectors = eig(A)

print values

print vectors[:, 0]
print np.dot(A, vectors[:, 0])