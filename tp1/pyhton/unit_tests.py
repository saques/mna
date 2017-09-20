import unittest
from lib import *
import scipy
import random

def gen_rand_square_matrix():
    """
    Generates a square random matrix
    :return: square matrix with random size (<=10) and random numbers (-50< <50)
    """
    n = int((random.random() * 987654321) % 10 + 1)
    return gen_rand_fixed_square_matrix(n)

def gen_rand_fixed_square_matrix(N):
    """
    Generates a square random matrix
    :return: square matrix with random size (<=10) and random numbers (-50< <50)
    """
    return np.multiply(np.subtract(np.random.random((N,N)), 0.5), 50).astype(int)

class LibTest(unittest.TestCase):
    def test_hess(self):
        A = gen_rand_square_matrix()
        H = scipy.linalg.hessenberg(A)
        H2 = np.asarray(hessemberg(A)[1])

        eql = np.isclose(H.flatten(), H2.flatten())
        for x in eql:
            if x == False:
                self.assertTrue(False, '\nMatrix that failed:\n{0}'.format(A))
        gen_rand_square_matrix()
        self.assertTrue(True)

    def test_eigh(self):
        rand = gen_rand_fixed_square_matrix(3)
        A = rand + rand.T

        L1, V1 = np.linalg.eigh(A)
        L2, V2 = eigh(A)

        print L1
        print V1
        print L2
        print V2

        eql = np.isclose(np.abs(L1.flatten()), np.abs(L2.flatten()))
        for x in eql:
            if x == False:
                self.assertTrue(False, '\nMatrix that failed:\n{0}'.format(A))
        gen_rand_square_matrix()
        self.assertTrue(True)

        eql = np.isclose(V1.flatten(), V2.flatten())
        for x in eql:
            if x == False:
                self.assertTrue(False, '\nMatrix that failed:\n{0}'.format(A))
        gen_rand_square_matrix()
        self.assertTrue(True)

