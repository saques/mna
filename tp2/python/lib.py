import numpy as np
import time


#Brute force implementation
def dft(x):
    N = len(x)
    ans = np.empty((N,)).astype(complex)

    for k in xrange(0, N):
        s = 0 + 0j
        for n in xrange(0, N):
            s += x[n]*np.exp((-2*np.pi*1j*n*k)/N)
        ans[k] = s
    np.exp
    return ans

#Cooley-Tukey implementation (only N = 2^j for some j)
def fft_ct_1(x, N):
    ans = None
    if N == 1:
        ans = np.empty((1,)).astype(complex)
        ans[0] = x[0]
    else:
        t1 = fft_ct_1(x[::2], N/2)
        t2 = fft_ct_1(x[1::2], N/2)
        ans = np.empty((N,)).astype(complex)
        for k in xrange(0, N/2):
            t = t1[k]
            ans[k] = t + np.exp((-2*np.pi*1j*k)/N)*t2[k]
            ans[k+N/2] = t - np.exp((-2*np.pi*1j*k)/N)*t2[k]
    return ans

def fft_ct(x):
    N = len(x)
    if not (N & (N-1)) == 0:
        raise ValueError("Length must be a power of 2")
    return fft_ct_1(x, N)


def ifft(x):
    N = len(x)

    if not (N & (N-1)) == 0:
        raise ValueError("Length must be a power of 2")

    A = bit_reverse(x).astype(complex)

    for s in range(1, 1+int(np.log(N)/np.log(2))):
        m = 1 << s
        Wm = np.exp(-2*np.pi*1j/m)
        for k in range(0, N, m):
            W = 1
            for j in range(0, m/2):
                t = W*A[k + j + m/2]
                u = A[k + j]
                A[k + j] = u + t
                A[k + j + m/2] = u - t
                W = W * Wm
    return A


def bit_reverse(x):
    N = len(x)
    if not (N & (N-1)) == 0:
        raise ValueError("Length must be a power of 2")
    ans = np.zeros((N,))
    ans[0] = x[0]
    ans[N-1] = x[N-1]
    bit_reverse_1(ans, x, N/2, 1, N/2, 1, N)
    return ans


def bit_reverse_1(ans, x, pos, val, n, inc, N):
    if pos >= N or pos <= 0 or n == 0:
        return
    ans[pos] = x[val]
    bit_reverse_1(ans, x, pos-n/2, val+inc, n/2, inc*2, N)
    bit_reverse_1(ans, x, pos+n/2, val+inc*2, n/2, inc*2, N)



